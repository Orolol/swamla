import os
import time
import random
import threading
from collections import deque
from typing import Any, Dict, List, Optional
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


class PackedFinewebDataset(IterableDataset):
    """
    Iterable dataset producing fixed-shape packed batches with minimal padding.
    - Always emits constant shape (batch_size, target_seq_len) for compile stability.
    - Packs multiple documents per sequence to maximize useful tokens/sec.
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 2048,
        batch_size: int = 4,
        buffer_docs: int = 4096,
        prefetch_batches: int = 16,
        shuffle: bool = True,
        tokenizer: Optional[Any] = None,
        num_workers: int = 1,
        start_offset: int = 0,
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length  # target_seq_len
        self.batch_size = batch_size
        self.buffer_docs = max(512, buffer_docs)
        self.prefetch_batches = max(4, prefetch_batches)
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers)

        # DDP awareness: detect if we're in a distributed environment
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                print(f"[PackedDataset] DDP detected: rank={self.rank}, world_size={self.world_size}")
            else:
                self.rank = 0
                self.world_size = 1
        except (ImportError, RuntimeError):
            self.rank = 0
            self.world_size = 1

        # For DDP: each rank skips to a different starting point to avoid data overlap
        effective_offset = start_offset + (self.rank * 1000)  # Offset each rank by 1000 examples

        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="CC-MAIN-2024-10",
            split=split,
            streaming=True,
        ).skip(effective_offset)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            access_token = os.getenv("HF_TOKEN")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)

        self.batch_queue = PackedBatchQueue(capacity=self.prefetch_batches)
        self.should_stop = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None
        self.exception: Optional[BaseException] = None

        self.stats = {
            "batches_built": 0,
            "docs_buffered": 0,
            "avg_padding_ratio": 0.0,
            "total_tokens": 0,
            "total_padding": 0,
        }

        self._closed = False

        # Register this dataset for cleanup
        _active_datasets.add(self)

        # Log DDP sharding info
        if self.world_size > 1:
            print(f"[PackedDataset Rank {self.rank}] Will process every {self.world_size}th example starting from offset {effective_offset}")

        self._start_producer()

    def _tokenize_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt", padding=False)
        ids = tokens["input_ids"].squeeze(0)
        if ids.numel() == 0 or ids[-1].item() != self.eos_id:
            # Ensure EOS at document end
            ids = torch.cat([ids, torch.tensor([self.eos_id], dtype=torch.long)])
        return ids.cpu()

    def _split_long(self, ids: torch.Tensor) -> List[torch.Tensor]:
        if ids.numel() <= self.max_length:
            return [ids]
        chunks = []
        start = 0
        L = ids.numel()
        while start < L:
            end = min(start + self.max_length, L)
            chunk = ids[start:end]
            if chunk[-1].item() != self.eos_id:
                chunk = torch.cat([chunk, torch.tensor([self.eos_id], dtype=torch.long)])
            if chunk.numel() > self.max_length:
                chunk = chunk[: self.max_length]
            chunks.append(chunk)
            start = end
        return chunks

    def _fill_sequence(self, docs: deque) -> torch.Tensor:
        # Returns 1D tensor length = max_length
        out = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
        pos = 0
        while docs and pos < self.max_length:
            cur = docs[0]
            remaining = self.max_length - pos
            if cur.numel() <= remaining:
                out[pos : pos + cur.numel()] = cur
                pos += cur.numel()
                docs.popleft()
            else:
                # Do not split the current document/chunk across sequences.
                # Finish this sequence (leave remaining padded) and keep the full chunk for the next sequence.
                break
        return out

    def _build_batch(self, docs_buffer: deque) -> Optional[Dict[str, torch.Tensor]]:
        if len(docs_buffer) == 0:
            return None

        input_ids = torch.full((self.batch_size, self.max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((self.batch_size, self.max_length), dtype=torch.long)

        for i in range(self.batch_size):
            seq = self._fill_sequence(docs_buffer)
            input_ids[i] = seq
            attention_mask[i] = (seq != self.pad_id).long()

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.pad_id
        labels[input_ids == self.pad_id] = -100

        total = input_ids.numel()
        used = attention_mask.sum().item()
        pad = total - used
        self.stats["total_tokens"] += total
        self.stats["total_padding"] += pad
        if self.stats["total_tokens"] > 0:
            self.stats["avg_padding_ratio"] = self.stats["total_padding"] / self.stats["total_tokens"]

        return {
            "input_ids": input_ids.contiguous(),  # [B, L]
            "attention_mask": attention_mask.contiguous(),  # [B, L]
            "decoder_input_ids": input_ids.clone().contiguous(),
            "decoder_attention_mask": attention_mask.clone().contiguous(),
            "labels": labels.contiguous(),
        }

    def _producer(self):
        try:
            it = iter(self.dataset)
            docs: List[torch.Tensor] = []
            docs_deque: deque = deque()
            example_count = 0  # Counter for DDP sharding

            while not self.should_stop.is_set():
                # Fill documents buffer
                while len(docs) < self.buffer_docs and not self.should_stop.is_set():
                    try:
                        ex = next(it)
                        example_count += 1

                        # DDP sharding: only process examples assigned to this rank
                        # This ensures each GPU gets different data without overlap
                        if self.world_size > 1:
                            if example_count % self.world_size != self.rank:
                                continue  # Skip this example, it belongs to another rank

                    except StopIteration:
                        it = iter(self.dataset)
                        example_count = 0  # Reset counter on dataset restart
                        continue
                    ids = self._tokenize_text(ex["text"])  # [T]
                    chunks = self._split_long(ids)
                    docs.extend(chunks)

                if self.shuffle and len(docs) > 0:
                    random.shuffle(docs)

                if len(docs_deque) == 0 and len(docs) > 0:
                    docs_deque = deque(docs)
                    docs = []

                if len(docs_deque) == 0:
                    time.sleep(0.01)
                    continue

                batch = self._build_batch(docs_deque)
                if batch is None:
                    continue
                if not self.batch_queue.put(batch):
                    break
                self.stats["batches_built"] += 1

        except BaseException as e:
            self.exception = e
        finally:
            self.batch_queue.close()

    def _start_producer(self):
        if self.producer_thread is not None and self.producer_thread.is_alive():
            return
        self.should_stop.clear()
        self.exception = None
        self.producer_thread = threading.Thread(target=self._producer, daemon=False, name="PackedProducer")
        self.producer_thread.start()

    def __iter__(self):
        self._start_producer()
        return self

    def __next__(self):
        if self.exception:
            raise self.exception
        batch = self.batch_queue.get()
        if batch is None:
            self.should_stop.set()
            raise StopIteration
        return batch

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def close(self):
        """Clean shutdown"""
        if self._closed:
            return  # Already closed
        
        self._closed = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Closing packed data loader")
        
        if hasattr(self, 'should_stop'):
            # Print stats before closing
            try:
                stats = self.get_stats()
                print(f"\n{'='*60}")
                print(f"Data Loader Final Statistics:")
                print(f"  - Batches built: {stats['batches_built']:,}")
                print(f"  - Documents buffered: {stats['docs_buffered']:,}")
                print(f"  - Average padding ratio: {stats['avg_padding_ratio']:.2%}")
                print(f"  - Total tokens: {stats['total_tokens']:,}")
                print(f"  - Total padding: {stats['total_padding']:,}")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Error printing stats: {e}")
            
            # Signal thread to stop
            self.should_stop.set()
            
            # Close the batch queue immediately to unblock any waiting threads
            if hasattr(self, "batch_queue"):
                self.batch_queue.close()
            
            # Give thread a moment to exit cleanly
            time.sleep(0.1)
            
            # Join thread with timeout
            if hasattr(self, 'producer_thread') and self.producer_thread is not None:
                if self.producer_thread.is_alive():
                    self.producer_thread.join(timeout=1.0)
                    if self.producer_thread.is_alive():
                        print(f"Warning: PackedProducer thread did not terminate")
            
            # Remove from active datasets
            _active_datasets.discard(self)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Packed data loader closed successfully")

    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during cleanup

