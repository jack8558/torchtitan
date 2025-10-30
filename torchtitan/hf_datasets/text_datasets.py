# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torch.utils import data  # <-- IMPORT ADDED

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

from dataflux_pytorch import dataflux_iterable_dataset
import json


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]

# --- MODIFIED: This is now a generator ---
def _process_gcs_text(sample: bytes):
    """
    Process GCS dataset sample bytes by decoding.
    This is now a GENERATOR that yields one text sample per JSON line.
    """
    try:
        decoded_string = sample.decode("utf-8")
        for line in decoded_string.strip().split("\n"):
            if line:
                data_dict = json.loads(line)
                yield data_dict["text"]
    except Exception as e:
        logger.warning(f"Failed to decode or parse JSON line: {e}. Skipping sample.")
        return

# --- MODIFIED: Added sort_listing_results=True ---
def _load_gcs_dataset(dataset_path: str):
    """Load GCS dataset with default configuration."""
    iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name="tpu-pytorch",
        bucket_name="torchprime",
        config=dataflux_iterable_dataset.Config(
            prefix=dataset_path,
            disable_compose=True,
            # CRITICAL: All ranks must read files in the same order
            # for sample-level sharding to be consistent.
            sort_listing_results=True,
        ),
    )
    return iterable_dataset



# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "gcs_c4_test": DatasetConfig(
        path="jackoh-exp/gcs-connector/c4_test",
        loader=partial(_load_gcs_dataset),
        sample_processor=_process_gcs_text,
    )
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )

        self.dataset_name = dataset_name

        # --- MODIFIED: Simplified GCS loading ---
        if dataset_name.startswith("gcs"):
            # Load the base DataFlux dataset.
            # We are NOT sharding here. All ranks get the same iterator.
            # Sharding will happen in __iter__.
            ds = dataset_loader(path)
            self._data = ds
        else:
            # Non-GCS datasets use the existing split_dataset_by_node
            ds = dataset_loader(path)
            self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor # This is now _process_gcs_text (a generator)

        # --- Store rank and world size ---
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        # ---------------------------------

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        
        # For DataFlux, this correctly calls its __iter__
        return iter(self._data)

    # --- ENTIRE __iter__ METHOD IS UPDATED ---
    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        # --- Get DataLoader worker info ---
        worker_info = data.get_worker_info()
        if worker_info is None:
            # Single-process loading (or main process)
            num_workers = 1
            worker_id = 0
        else:
            # Multi-process loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # --- Calculate global rank for sample sharding ---
        # This combines DDP rank and DataLoader worker rank
        # to give every single process a unique ID.
        global_rank = self.dp_rank * num_workers + worker_id
        global_world_size = self.dp_world_size * num_workers

        logger.info(
            f"[Rank {self.dp_rank} (Worker {worker_id})] "
            f"Starting iter. Global Rank: {global_rank} / {global_world_size}"
        )

        is_gcs_dataset = self.dataset_name.startswith("gcs")

        while True:
            overall_sample_index = 0
            samples_processed_by_this_rank = 0

            # self._get_data_iter() returns an iterator
            # For GCS, 'sample_or_file_bytes' is the raw bytes of one file
            # For HF, 'sample_or_file_bytes' is one pre-sharded sample
            for sample_or_file_bytes in self._get_data_iter():

                if is_gcs_dataset:
                    # --- GCS SHARDING LOGIC ---
                    
                    # _text_processor is _process_gcs_text (our generator)
                    # It yields one 'sample_text' (one JSON line) at a time
                    for sample_text in self._text_processor(sample_or_file_bytes):
                        
                        # --- THIS IS THE MANUAL SHARDING ---
                        # Each process only handles samples where the index
                        # matches its unique global rank.
                        if overall_sample_index % global_world_size == global_rank:
                            if not sample_text:
                                continue # Skip empty samples

                            sample_tokens = self._tokenizer.encode(
                                sample_text, add_bos=True, add_eos=True
                            )
                            self._token_buffer.extend(sample_tokens)
                            samples_processed_by_this_rank += 1

                            while len(self._token_buffer) >= max_buffer_token_len:
                                x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                                self._token_buffer = self._token_buffer[max_buffer_token_len:]
                                input = x[:-1]
                                label = x[1:]
                                yield {"input": input}, label
                        
                        # This index must increment for *every* sample,
                        # even those skipped by other ranks.
                        overall_sample_index += 1
                
                else:
                    # --- Original HuggingFace Logic ---
                    # Data is already sharded by split_dataset_by_node
                    sample_text = self._text_processor(sample_or_file_bytes)
                    sample_tokens = self._tokenizer.encode(
                        sample_text, add_bos=True, add_eos=True
                    )
                    self._token_buffer.extend(sample_tokens)
                    samples_processed_by_this_rank += 1

                    while len(self._token_buffer) >= max_buffer_token_len:
                        x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                        self._token_buffer = self._token_buffer[max_buffer_token_len:]
                        input = x[:-1]
                        label = x[1:]
                        yield {"input": input}, label

            # --- End of all files/samples ---
            log_prefix = f"[Rank {self.dp_rank} (Worker {worker_id})]"
            log_msg = (
                f"(Processed {samples_processed_by_this_rank} samples)"
            )

            if not self.infinite:
                logger.info(f"========== {log_prefix} {log_msg} ==========")
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                logger.info(f"========== {log_prefix} {log_msg} ==========")
                
                # Reset counters for next loop
                self._sample_idx = 0 
                samples_processed_by_this_rank = 0

                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            # This handles both HF streaming datasets and our
            # DataFluxIterableDataset (if it were to implement state_dict)
            if hasattr(self._data, "load_state_dict") and "data" in state_dict:
                self._data.load_state_dict(state_dict["data"])
            else:
                logger.warning("Could not load state_dict for iterable dataset.")


    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            if hasattr(self._data, "state_dict"):
                _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        # num_workers=... (if you add num_workers > 0 here,
        # the worker sharding in __iter__ will activate)
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )