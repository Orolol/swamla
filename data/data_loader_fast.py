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


class FastBatchQueue:
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


class FastFinewebDataset(IterableDataset):
    """
    Iterable dataset producing fixed-shape UNPACKED batches with high throughput.
    - Uses a producer thread for prefetching (like PackedFinewebDataset).
    - Does NOT pack sequences: 1 document = 1 sequence (padded).
    - Splits long documents into chunks.
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
        fixed_length: bool = True,  # Pad to max_length for torch.compile compatibility
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_docs = max(512, buffer_docs)
        self.prefetch_batches = max(4, prefetch_batches)
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers)
        self.fixed_length = fixed_length

        # DDP awareness
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                print(f"[FastDataset] DDP detected: rank={self.rank}, world_size={self.world_size}")
            else:
                self.rank = 0
                self.world_size = 1
        except (ImportError, RuntimeError):
            self.rank = 0
            self.world_size = 1

        effective_offset = start_offset + (self.rank * 1000)

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

        self.batch_queue = FastBatchQueue(capacity=self.prefetch_batches)
        self.should_stop = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None
        self.exception: Optional[BaseException] = None

        self.stats = {
            "batches_built": 0,
            "docs_buffered": 0,
            "total_tokens": 0,
            "total_padding": 0,
        }

        self._closed = False
        _active_datasets.add(self)

        if self.world_size > 1:
            print(f"[FastDataset Rank {self.rank}] Will process every {self.world_size}th example starting from offset {effective_offset}")

        self._start_producer()

    def _tokenize_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt", padding=False)
        ids = tokens["input_ids"].squeeze(0)
        if ids.numel() == 0 or ids[-1].item() != self.eos_id:
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

    def _build_batch(self, batch_docs: List[torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        if len(batch_docs) < self.batch_size:
            return None

        # Determine sequence length for this batch
        if self.fixed_length:
            # Fixed padding: always use max_length for torch.compile compatibility
            batch_seq_len = self.max_length
        else:
            # Dynamic padding: use max length in this batch (causes recompilation)
            max_len_in_batch = max(len(d) for d in batch_docs)
            batch_seq_len = min(max_len_in_batch, self.max_length)

        input_ids = torch.full((self.batch_size, batch_seq_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((self.batch_size, batch_seq_len), dtype=torch.long)

        for i, doc in enumerate(batch_docs):
            length = min(doc.numel(), batch_seq_len)
            input_ids[i, :length] = doc[:length]
            attention_mask[i, :length] = 1

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.pad_id
        labels[input_ids == self.pad_id] = -100

        total = input_ids.numel()
        used = attention_mask.sum().item()
        pad = total - used
        self.stats["total_tokens"] += total
        self.stats["total_padding"] += pad

        return {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels.contiguous(),
        }

    def _producer(self):
        try:
            it = iter(self.dataset)
            docs_buffer: List[torch.Tensor] = []
            example_count = 0
            
            print(f"[FastDataset] Starting Smart Batching Producer (Buffer: {self.buffer_docs})...")

            while not self.should_stop.is_set():
                # 1. Fill Buffer
                # We need enough docs to form several batches to make sorting effective
                while len(docs_buffer) < self.buffer_docs and not self.should_stop.is_set():
                    try:
                        ex = next(it)
                        example_count += 1
                        if self.world_size > 1 and example_count % self.world_size != self.rank:
                            continue
                    except StopIteration:
                        it = iter(self.dataset)
                        example_count = 0
                        continue
                    
                    ids = self._tokenize_text(ex["text"])
                    chunks = self._split_long(ids)
                    docs_buffer.extend(chunks)
                
                if self.should_stop.is_set():
                    break

                # 2. Smart Batching Strategy
                # Sort by length to group similar sized sequences
                # This minimizes padding within each batch
                if self.shuffle:
                    # Add a tiny bit of noise to length to avoid deterministic ordering of identical lengths?
                    # For now, simple sort is fine.
                    docs_buffer.sort(key=lambda x: len(x))
                
                # 3. Create Batches
                batches = []
                while len(docs_buffer) >= self.batch_size:
                    # Take the next batch_size elements (which are similar in length)
                    batch_docs = [docs_buffer.pop(0) for _ in range(self.batch_size)]
                    batches.append(batch_docs)
                
                # 4. Shuffle Batches
                # We want to yield batches in random order to preserve stochasticity
                if self.shuffle:
                    random.shuffle(batches)
                
                # 5. Enqueue Batches
                for b_docs in batches:
                    if self.should_stop.is_set():
                        break
                    
                    batch = self._build_batch(b_docs)
                    if batch is not None:
                        if not self.batch_queue.put(batch):
                            self.should_stop.set()
                            break
                        self.stats["batches_built"] += 1
                
                # Note: Leftover docs ( < batch_size) remain in docs_buffer for next round

        except BaseException as e:
            self.exception = e
            print(f"[FastDataset] Producer exception: {e}")
        finally:
            self.batch_queue.close()

    def _start_producer(self):
        if self.producer_thread is not None and self.producer_thread.is_alive():
            return
        self.should_stop.clear()
        self.exception = None
        self.producer_thread = threading.Thread(target=self._producer, daemon=False, name="FastProducer")
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
        if self._closed:
            return
        
        self._closed = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Closing fast data loader")
        
        if hasattr(self, 'should_stop'):
            self.should_stop.set()
            if hasattr(self, "batch_queue"):
                self.batch_queue.close()
            time.sleep(0.1)
            if hasattr(self, 'producer_thread') and self.producer_thread is not None:
                if self.producer_thread.is_alive():
                    self.producer_thread.join(timeout=1.0)
            
            _active_datasets.discard(self)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fast data loader closed successfully")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass