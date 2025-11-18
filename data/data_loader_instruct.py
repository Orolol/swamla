"""
Instruction fine-tuning data loader for SlimOrca dataset.
Converts conversations to ChatML format with loss masking on prompts.
"""

import os
import time
import random
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import signal
import atexit
import weakref
from datetime import datetime

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Global registry to track active datasets for cleanup
_active_datasets = weakref.WeakSet()


def _cleanup_all_datasets(signum=None, frame=None):
    """Signal handler to clean up all active datasets"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received interrupt signal, cleaning up datasets...")
    for dataset in list(_active_datasets):
        try:
            dataset.close()
        except Exception as e:
            print(f"Error closing dataset: {e}")

    # Re-raise KeyboardInterrupt for proper exit
    if signum == signal.SIGINT:
        raise KeyboardInterrupt


# Register signal handlers
signal.signal(signal.SIGINT, _cleanup_all_datasets)
signal.signal(signal.SIGTERM, _cleanup_all_datasets)
atexit.register(_cleanup_all_datasets)


class PackedBatchQueue:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.queue: deque = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.closed = False

    def put(self, item: Dict[str, torch.Tensor]) -> bool:
        with self.lock:
            while len(self.queue) >= self.capacity and not self.closed:
                self.not_full.wait(timeout=1.0)
                if self.closed:
                    return False
            if self.closed:
                return False
            self.queue.append(item)
            self.not_empty.notify()
            return True

    def get(self) -> Optional[Dict[str, torch.Tensor]]:
        with self.lock:
            while len(self.queue) == 0 and not self.closed:
                self.not_empty.wait(timeout=1.0)
                if self.closed and len(self.queue) == 0:
                    return None
            if len(self.queue) == 0:
                return None
            item = self.queue.popleft()
            self.not_full.notify()
            return item

    def close(self):
        with self.lock:
            self.closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def __len__(self):
        with self.lock:
            return len(self.queue)


class PackedInstructDataset(IterableDataset):
    """
    Iterable dataset for instruction fine-tuning with SlimOrca.
    - Converts conversations to ChatML format
    - Packs multiple conversations into sequences
    - Provides loss masking (loss only on assistant responses)
    - Always emits constant shape (batch_size, max_length) for compile stability
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 2048,
        batch_size: int = 4,
        buffer_docs: int = 100,  # Reduced from 2048 - number of conversations to buffer before building batch
        prefetch_batches: int = 16,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        tokenizer: Optional[Any] = None,
        num_workers: int = 1,
        start_offset: int = 0,
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_docs = buffer_docs  # Number of conversations to buffer before building a batch
        self.prefetch_batches = prefetch_batches  # Number of batches to prefetch
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = max(1, num_workers)
        self.start_offset = start_offset

        # DDP awareness
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                print(f"[InstructDataset] DDP detected: rank={self.rank}, world_size={self.world_size}")
            else:
                self.rank = 0
                self.world_size = 1
        except (ImportError, RuntimeError):
            self.rank = 0
            self.world_size = 1

        # For DDP: each rank skips to a different starting point
        effective_offset = start_offset + (self.rank * 500)

        # Load SlimOrca dataset
        self.dataset = load_dataset(
            "Open-Orca/SlimOrca",
            split=split,
            streaming=True,
        )

        # Apply shuffle BEFORE skip
        if self.shuffle:
            shuffle_seed = start_offset + self.rank
            print(f"[InstructDataset Rank {self.rank}] Shuffling dataset with seed={shuffle_seed}, buffer_size={shuffle_buffer_size}")
            self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)

        # Skip to starting position AFTER shuffle
        if effective_offset > 0:
            print(f"[InstructDataset Rank {self.rank}] Skipping to offset {effective_offset}")
            self.dataset = self.dataset.skip(effective_offset)

        # Setup tokenizer with ChatML tokens
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add ChatML special tokens if not already present
        special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0 and self.rank == 0:
            print(f"Added {num_added} special tokens to tokenizer (ChatML format)")

        # Get token IDs for special tokens
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)

        # Cache commonly used tokenized sequences to avoid redundant tokenization
        self._token_cache = {}
        for role in ["system", "user", "assistant"]:
            header = f"<|im_start|>{role}\n"
            self._token_cache[f"header_{role}"] = self.tokenizer(header, add_special_tokens=False)['input_ids']
        self._token_cache["closing"] = self.tokenizer("<|im_end|>", add_special_tokens=False)['input_ids']
        self._token_cache["newline"] = self.tokenizer("\n", add_special_tokens=False)['input_ids']

        self.batch_queue = PackedBatchQueue(capacity=self.prefetch_batches)
        self.should_stop = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None
        self.exception: Optional[BaseException] = None

        self.stats = {
            "batches_built": 0,
            "conversations_packed": 0,
            "avg_padding_ratio": 0.0,
            "avg_mask_ratio": 0.0,  # % of tokens with loss calculated
            "total_tokens": 0,
            "total_padding": 0,
            "total_masked": 0,
        }

        self._closed = False

        # Register this dataset for cleanup
        _active_datasets.add(self)

    def convert_to_chatml(self, conversation: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Convert a SlimOrca conversation to ChatML format with loss masking.
        OPTIMIZED:
        - Tokenizes each message separately to track loss mask directly
        - Uses cached tokens for headers and closing tags
        - Avoids expensive offset_mapping computation

        Args:
            conversation: List of messages with 'from' and 'value' keys
                         'from' can be: 'system', 'human', 'gpt'

        Returns:
            Tuple of (token_ids, loss_mask) where loss_mask[i] = 1 means calculate loss
        """
        all_token_ids = []
        all_loss_masks = []

        for msg in conversation:
            role = msg.get("from", "")
            content = msg.get("value", "")

            # Skip empty messages
            if not content or not role:
                continue

            # Map SlimOrca roles to ChatML roles
            if role == "system":
                chatml_role = "system"
            elif role == "human":
                chatml_role = "user"
            elif role == "gpt":
                chatml_role = "assistant"
            else:
                # Unknown role, skip
                continue

            # Use cached header tokens
            header_ids = self._token_cache[f"header_{chatml_role}"].copy()

            # Tokenize content (this is the only part that varies)
            content_ids = self.tokenizer(content, add_special_tokens=False)['input_ids']

            # Use cached closing tokens
            closing_ids = self._token_cache["closing"].copy()

            # Combine tokens for this message
            msg_tokens = header_ids + content_ids + closing_ids

            # Create loss mask: only train on assistant content (not header/closing)
            if chatml_role == "assistant":
                # Mask: 0 for header, 1 for content, 0 for closing
                msg_mask = ([0] * len(header_ids) +
                           [1] * len(content_ids) +
                           [0] * len(closing_ids))
            else:
                # Don't train on system/user messages
                msg_mask = [0] * len(msg_tokens)

            all_token_ids.extend(msg_tokens)
            all_loss_masks.extend(msg_mask)

        # Add final newline token (cached)
        sep_ids = self._token_cache["newline"].copy()
        all_token_ids.extend(sep_ids)
        all_loss_masks.extend([0] * len(sep_ids))

        return all_token_ids, all_loss_masks

    def _producer_loop(self):
        """Producer thread that builds packed batches."""
        import time

        timings = {
            'dataset_iteration': 0,
            'chatml_conversion': 0,
            'batch_building': 0,
            'queue_put': 0,
        }
        batch_count = 0
        conversations_processed = 0

        try:
            buffer_tokens = []
            buffer_masks = []

            for example in self.dataset:
                t0 = time.perf_counter()

                if self.should_stop.is_set():
                    break

                timings['dataset_iteration'] += time.perf_counter() - t0
                t0 = time.perf_counter()

                # Convert conversation to ChatML
                try:
                    # SlimOrca format: example['conversations'] is a list of messages
                    conversation = example.get('conversations', [])
                    if not conversation:
                        continue

                    tokens, mask = self.convert_to_chatml(conversation)

                    # Skip if too long, empty, or too short
                    # Minimum of 10 tokens ensures we have at least some content
                    if len(tokens) < 10 or len(tokens) > self.max_length:
                        continue

                    # Skip if no assistant responses (no tokens to train on)
                    if sum(mask) == 0:
                        continue

                    buffer_tokens.append(tokens)
                    buffer_masks.append(mask)
                    conversations_processed += 1

                    timings['chatml_conversion'] += time.perf_counter() - t0

                except Exception as e:
                    if self.rank == 0:
                        print(f"[InstructDataset] Error processing conversation: {e}")
                    continue

                # When buffer is full enough, build a batch
                if len(buffer_tokens) >= self.buffer_docs:
                    t0 = time.perf_counter()
                    batch = self._build_batch(buffer_tokens, buffer_masks)
                    timings['batch_building'] += time.perf_counter() - t0

                    if batch is not None:
                        t0 = time.perf_counter()
                        success = self.batch_queue.put(batch)
                        timings['queue_put'] += time.perf_counter() - t0

                        batch_count += 1

                        # Log timing every 10 batches
                        if self.rank == 0 and batch_count % 10 == 0:
                            total_time = sum(timings.values())
                            # print(f"[DataLoader] Batch {batch_count} timing - {conversations_processed} convs in {total_time:.2f}s:")
                            # if conversations_processed > 0:
                            #     print(f"  Dataset iter:  {timings['dataset_iteration']*1000/conversations_processed:.1f}ms/conv ({timings['dataset_iteration']/total_time*100:.1f}%)")
                            #     print(f"  ChatML conv:   {timings['chatml_conversion']*1000/conversations_processed:.1f}ms/conv ({timings['chatml_conversion']/total_time*100:.1f}%)")
                            #     print(f"  Batch build:   {timings['batch_building']*1000/10:.1f}ms/batch ({timings['batch_building']/total_time*100:.1f}%)")
                            #     print(f"  Queue put:     {timings['queue_put']*1000/10:.1f}ms/batch ({timings['queue_put']/total_time*100:.1f}%)")
                            #     print(f"  Throughput:    {conversations_processed/total_time:.1f} convs/sec")
                            # Reset timings
                            timings = {k: 0 for k in timings}
                            conversations_processed = 0

                        if not success:
                            break

                    # Clear buffer
                    buffer_tokens = []
                    buffer_masks = []

            # Process remaining conversations
            if buffer_tokens and not self.should_stop.is_set():
                batch = self._build_batch(buffer_tokens, buffer_masks)
                if batch is not None:
                    self.batch_queue.put(batch)

        except Exception as e:
            self.exception = e
            if self.rank == 0:
                print(f"[InstructDataset] Producer thread error: {e}")
        finally:
            self.batch_queue.close()

    def _build_batch(self, buffer_tokens: List[List[int]], buffer_masks: List[List[int]]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build a packed batch from buffered conversations.

        Returns:
            Dict with keys: 'input_ids', 'loss_mask'
            Both have shape (batch_size, max_length)
        """
        if not buffer_tokens:
            return None

        batch_input_ids = []
        batch_loss_masks = []

        for _ in range(self.batch_size):
            seq_tokens = []
            seq_mask = []

            # Pack conversations into this sequence
            while buffer_tokens and len(seq_tokens) < self.max_length:
                conv_tokens = buffer_tokens[0]
                conv_mask = buffer_masks[0]

                # Check if conversation fits
                if len(seq_tokens) + len(conv_tokens) <= self.max_length:
                    seq_tokens.extend(conv_tokens)
                    seq_mask.extend(conv_mask)
                    buffer_tokens.pop(0)
                    buffer_masks.pop(0)
                    self.stats["conversations_packed"] += 1
                else:
                    # Can't fit, try next sequence
                    break

            # Skip sequences that are too short (all or almost all padding)
            # Require at least 10 tokens of actual content
            if len(seq_tokens) < 10:
                continue

            # Pad to max_length
            padding_length = self.max_length - len(seq_tokens)
            if padding_length > 0:
                seq_tokens.extend([self.pad_id] * padding_length)
                seq_mask.extend([0] * padding_length)  # Don't calculate loss on padding
                self.stats["total_padding"] += padding_length

            batch_input_ids.append(seq_tokens)
            batch_loss_masks.append(seq_mask)

            # Update stats
            self.stats["total_tokens"] += self.max_length
            self.stats["total_masked"] += sum(seq_mask)

        # Skip this batch if no valid sequences were created
        if len(batch_input_ids) == 0:
            return None

        # Convert to tensors
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        loss_mask = torch.tensor(batch_loss_masks, dtype=torch.float)

        # Update stats
        self.stats["batches_built"] += 1
        padding_ratio = self.stats["total_padding"] / max(1, self.stats["total_tokens"])
        mask_ratio = self.stats["total_masked"] / max(1, self.stats["total_tokens"])
        self.stats["avg_padding_ratio"] = padding_ratio
        self.stats["avg_mask_ratio"] = mask_ratio

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

    def __iter__(self):
        """Start producer thread and yield batches."""
        if self._closed:
            raise RuntimeError("Cannot iterate over closed dataset")

        # Start producer thread
        self.should_stop.clear()
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()

        # Yield batches from queue
        while True:
            batch = self.batch_queue.get()
            if batch is None:
                break

            # Check for producer exceptions
            if self.exception is not None:
                raise self.exception

            yield batch

        # Wait for producer thread to finish
        if self.producer_thread is not None:
            self.producer_thread.join(timeout=5.0)

    def close(self):
        """Stop producer thread and close dataset."""
        if self._closed:
            return

        self._closed = True
        self.should_stop.set()
        self.batch_queue.close()

        if self.producer_thread is not None and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5.0)

        if self.rank == 0:
            print(f"\n[InstructDataset] Stats:")
            print(f"  Batches built: {self.stats['batches_built']}")
            print(f"  Conversations packed: {self.stats['conversations_packed']}")
            print(f"  Avg padding ratio: {self.stats['avg_padding_ratio']:.2%}")
            print(f"  Avg mask ratio (loss tokens): {self.stats['avg_mask_ratio']:.2%}")

    def __del__(self):
        self.close()


def test_instruct_loader():
    """Test the instruction data loader."""
    print("Testing PackedInstructDataset...")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Add ChatML tokens
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    tokenizer.add_special_tokens(special_tokens)

    # Create dataset
    dataset = PackedInstructDataset(
        split="train",
        max_length=512,
        batch_size=2,
        buffer_docs=100,
        shuffle=True,
        tokenizer=tokenizer,
    )

    # Get a few batches
    print("\nFetching 3 batches...\n")
    for i, batch in enumerate(dataset):
        if i >= 3:
            break

        print(f"Batch {i+1}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  loss_mask shape: {batch['loss_mask'].shape}")
        print(f"  loss_mask sum: {batch['loss_mask'].sum().item()} / {batch['loss_mask'].numel()} tokens")
        print(f"  loss_mask ratio: {batch['loss_mask'].sum().item() / batch['loss_mask'].numel():.2%}")

        # Decode first sequence to verify format
        first_seq = batch['input_ids'][0]
        first_mask = batch['loss_mask'][0]

        print(f"\n  First sequence preview:")
        decoded = tokenizer.decode(first_seq[:200])
        print(f"  {decoded[:500]}...")

        # Show which parts have loss calculated
        print(f"\n  Loss mask preview (first 50 tokens):")
        for j in range(min(50, len(first_seq))):
            token = tokenizer.decode([first_seq[j]])
            mask = int(first_mask[j].item())
            if mask == 1:
                print(f"    [{j}] '{token}' -> LOSS")

        print("\n" + "="*80 + "\n")

    dataset.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_instruct_loader()
