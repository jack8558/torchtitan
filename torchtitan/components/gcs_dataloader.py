# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


import torch
from dataflux_pytorch import dataflux_iterable_dataset

def build_gcs_dataloader(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: train_spec_module.BaseTokenizer,
) -> torch.utils.data.DataLoader:
    """
    Builds a PyTorch DataLoader for a dataset stored in Google Cloud Storage.

    This function uses the dataflux-pytorch library to stream data from GCS
    as an IterableDataset.

    Args:
        job_config: The main job configuration.
        dp_world_size: The world size of the data parallel group.
        dp_rank: The rank of the current process in the data parallel group.
        tokenizer: The tokenizer to use for preprocessing.

    Returns:
        A torch.utils.data.DataLoader instance.
    """
    # The Llama tokenizer does not have a pad_id, so we need to set it to the eos_id.
    # This is a common practice for models that don't have a specific padding token.
    if getattr(tokenizer, "pad_id", None) is None:
        tokenizer.pad_id = tokenizer.eos_id

    gcs_config = job_config.gcs_dataset
    print(f"Connecting to GCS: gs://{gcs_config.bucket_name}/{gcs_config.data_prefix}")

    iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name=gcs_config.project_id,
        bucket_name=gcs_config.bucket_name,
        config=dataflux_iterable_dataset.Config(
            prefix=gcs_config.data_prefix,
            disable_compose=True,
        ),
    )

    # AI generated code
    def collate_fn(batch_of_byte_samples):
        tokenized_samples = []
        for byte_sample in batch_of_byte_samples:
            try:
                if not byte_sample:
                    continue
                text = byte_sample.decode("utf-8")
                # Tokenize the text
                tokens = tokenizer.encode(text, bos=True, eos=True)
                tokenized_samples.append(torch.tensor(tokens, dtype=torch.long))
            except (UnicodeDecodeError, IndexError):
                # Skip corrupted data
                continue

        if not tokenized_samples:
            # If all samples in the batch were bad, return None to skip.
            return None

        # Pad to the longest sequence in the batch
        padded_tokens = torch.nn.utils.rnn.pad_sequence(tokenized_samples, batch_first=True, padding_value=tokenizer.pad_id)

        inputs = padded_tokens[:, :-1]
        labels = padded_tokens[:, 1:]

        return [{"input": inputs}, labels]

    return torch.utils.data.DataLoader(
        iterable_dataset,
        batch_size=job_config.training.local_batch_size,
        collate_fn=collate_fn,
    )