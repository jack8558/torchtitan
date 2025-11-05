# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

# --- GCS MODIFICATION START ---
# We need gcsfs for file system operations (ls, rm, isdir)
# and to implicitly handle dcp.save/load via fsspec
try:
    import gcsfs
    import fsspec

    _GCSFS_AVAILABLE = True
except ImportError:
    _GCSFS_AVAILABLE = False
# --- GCS MODIFICATION END ---

from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig, TORCH_DTYPE_MAP
from torchtitan.protocols import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


# --- GCS MODIFICATION START ---
# Updated purge_thread to accept an fsspec filesystem object
def purge_thread(purge_queue: queue.Queue, fs: fsspec.AbstractFileSystem | None = None):
    # --- GCS MODIFICATION END ---
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
        # --- GCS MODIFICATION ---
        fs (fsspec.AbstractFileSystem | None): The filesystem object (e.g., gcsfs) to use for deletion.
                                              If None, use local filesystem.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            
            # --- GCS MODIFICATION START ---
            try:
                if fs:
                    # Use fsspec's recursive remove for GCS
                    fs.rm(path, recursive=True)
                else:
                    # Original local filesystem logic
                    shutil.rmtree(path, ignore_errors=True)
                logger.info(
                    "Checkpointer deleted %s in %.2f seconds.",
                    path,
                    time.monotonic() - begin,
                )
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {path}: {e}")
            # --- GCS MODIFICATION END ---
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.
    ... (rest of docstring) ...
    """

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ) -> None:
        self.enable = checkpoint_config.enable
        self.load_only = checkpoint_config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.ft_manager = (
            ft_manager.manager if ft_manager and ft_manager.enabled else None
        )

        self.enable_ft_dataloader_checkpoints = (
            self.ft_manager and checkpoint_config.enable_ft_dataloader_checkpoints
        )

        if self.ft_manager and not self.enable_ft_dataloader_checkpoints:
            logger.warn(
                "Fault tolerance is enabled but enable_ft_dataloader_checkpoints is False. "
                "This means replicas can retrain over the same data multiple times, which can result in overfitting."
            )

        if self.ft_manager:
            optimizers.init_cache_state_dict()

            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {
                        MODEL,
                        OPTIMIZER,
                        LR_SCHEDULER,
                        TRAIN_STATE,
                    }:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
            self.ft_replica_id = ft_manager.replica_id

        async_mode = checkpoint_config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ) or self.enable_ft_dataloader_checkpoints

        if not self.enable and not self.enable_ft_dataloader_checkpoints:
            return

        self.ft_states = {DATALOADER: dataloader}

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None

        # --- GCS MODIFICATION START ---
        # Logic to handle GCS paths and initialize fsspec filesystem
        self.fs: fsspec.AbstractFileSystem | None = None
        self.is_gcs = False
        self.sep = os.path.sep

        base_path = base_folder or ""
        folder_path = checkpoint_config.folder

        # Determine if we are using GCS and set the full folder path
        if folder_path.startswith("gs://"):
            self.folder = folder_path.rstrip("/")
            self.is_gcs = True
        elif base_path.startswith("gs://"):
            self.folder = f"{base_path.rstrip('/')}/{folder_path.lstrip('/')}"
            self.is_gcs = True
        else:
            self.folder = os.path.join(base_path, folder_path)

        if self.is_gcs:
            if not _GCSFS_AVAILABLE:
                raise ImportError(
                    "GCS path detected, but 'gcsfs' is not installed. "
                    "Please install with 'pip install gcsfs'"
                )
            logger.info(
                "GCS path detected. Using 'gcsfs' for all checkpointing operations."
            )
            self.fs = gcsfs.GCSFileSystem()
            self.sep = self.fs.sep
        # --- GCS MODIFICATION END ---

        # Checkpoint policy related fields.
        self.initial_load_model_only = checkpoint_config.initial_load_model_only
        self.initial_load_in_hf = checkpoint_config.initial_load_in_hf
        self.initial_load_path = checkpoint_config.initial_load_path
        self.initial_load_in_hf_quantized = (
            checkpoint_config.initial_load_in_hf_quantized
        )
        self.last_save_model_only = checkpoint_config.last_save_model_only
        self.last_save_in_hf = checkpoint_config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                sd_adapter is not None
            ), "job_config.checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
        self.sd_adapter = sd_adapter
        self.export_dtype = TORCH_DTYPE_MAP[checkpoint_config.export_dtype]
        self.exclude_from_loading = checkpoint_config.exclude_from_loading
        self.interval = checkpoint_config.interval
        self.enable_first_step_checkpoint = (
            checkpoint_config.enable_first_step_checkpoint
        )

        # Async checkpoint related fields.
        async_mode = checkpoint_config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            or self.enable_ft_dataloader_checkpoints
        ):
            self.pg = dist.new_group(backend="gloo")

        self.keep_latest_k = checkpoint_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            # --- GCS MODIFICATION START ---
            # Pass the filesystem object to the purge thread
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue, self.fs), daemon=True
            )
            # --- GCS MODIFICATION END ---
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(
                f"Unknown checkpoint async_mode {checkpoint_config.async_mode}"
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | None = None
        
        # --- GCS MODIFICATION START ---
        # Use GCS-aware path join
        join = self.sep.join
        # --- GCS MODIFICATION END ---

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        if to_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    # --- GCS MODIFICATION ---
                    path=join([checkpoint_id, "sharded"]),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # ... (original code) ...
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and self.sd_adapter.fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                # --- GCS MODIFICATION ---
                input_dir=join([checkpoint_id, "sharded"]),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=self.sd_adapter.fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(
                hf_state_dict,
                storage_reader=hf_storage_reader,
            )

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            # --- GCS MODIFICATION ---
            # This will use the implicit fsspec/gcsfs path if checkpoint_id is "gs://"
            # No dataflux logic is needed for this test.
            dcp.load(state_dict, checkpoint_id=checkpoint_id)
            # --- GCS MODIFICATION END ---

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the checkpoint for the current step.
        ... (rest of docstring) ...
        """

        if self.enable_ft_dataloader_checkpoints:
            self.ft_save(curr_step)

        if not self._should_save(curr_step, last_step):
            return

        begin = time.monotonic()
        if not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager and self.ft_manager.participating_rank() == 0
        ):
            logger.info("Saving the checkpoint (or staging if async is enabled).")
            checkpoint_id = self._create_checkpoint_id(curr_step)
            self._async_wait()
            # This GC is called for async checkpoint as it is useless to do
            # GC right after async_save -- the CPU memory is not able to be
            # freed until _async_wait()
            if last_step:
                self._save_last_step(curr_step)
                return

            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                    # --- FIX: Pass to_hf=False ---
                    to_hf=False,
                )
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
                self.staging = True
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = self.dcp_save(
                    states, 
                    checkpoint_id=checkpoint_id, 
                    async_mode=self.async_mode,
                    # --- FIX: Pass to_hf=False ---
                    to_hf=False,
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                    # --- FIX: Pass to_hf=False ---
                    to_hf=False,
                )
            self._purge_stale_checkpoints()

            logger.info(
                "Finished saving the checkpoint (or staging if async is enabled)"
                f"in {time.monotonic() - begin:.2f} seconds."
            )
        elif self.enable_ft_dataloader_checkpoints:
            assert self.ft_manager is not None
            logger.info(
                "Replica %d doesn't save checkpoint.",
                self.ft_manager.participating_rank(),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.
        ... (rest of docstring) ...
        """

        if self.enable_ft_dataloader_checkpoints:
            self._ft_load()

        if not self.enable:
            return False
            
        # --- GCS MODIFICATION START ---
        # Use fsspec to check for existence of folder
        isdir = self.fs.isdir if self.is_gcs else os.path.isdir
        # --- GCS MODIFICATION END ---

        model_only = False
        from_hf = False
        from_quantized = False
        
        # --- GCS MODIFICATION ---
        # Original: if not os.path.exists(self.folder):
        if not isdir(self.folder):
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized
            if from_hf:
                assert (
                    model_only
                ), "Only model can be loaded when loading from HF's safetensors checkpoint."

            if from_quantized:
                assert (
                    from_hf
                ), "Quantized checkpoint can only be loaded from HuggingFace format."

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                # --- GCS MODIFICATION ---
                if not isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                if from_hf:
                    logger.info(
                        f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                    )
            elif from_hf:
                checkpoint_id = self.sd_adapter.hf_assets_path
                # --- GCS MODIFICATION ---
                if not isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights but the path is not valid. \
                        Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"loading HF safetensors from --model.hf_assets_path: {self.sd_adapter.hf_assets_path}"
                )
            else:
                return False
        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                    f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
                )
            if self.initial_load_in_hf:
                logger.warning(
                    "checkpoint.initial_load_in_hf is True but the checkpoint.folder exists. "
                    "Checkpointer will not load from HF safetensors"
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            # --- GCS MODIFICATION ---
            if not isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.
        ... (rest of docstring) ...
        """
        if self.enable_staging and self.staging:
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.
        ... (rest of docstring) ...
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        # --- GCS MODIFICATION START ---
        # Use fsspec-aware helpers for GCS paths
        join = self.sep.join
        isdir = self.fs.isdir if self.is_gcs else os.path.isdir
        isfile = self.fs.isfile if self.is_gcs else os.path.isfile

        if not isdir(folder):
            return -1

        try:
            # Use fsspec to list directory contents
            filenames = (
                [f.split(self.sep)[-1] for f in self.fs.ls(folder, detail=False)]
                if self.is_gcs
                else os.listdir(folder)
            )
        except FileNotFoundError:
            return -1
        # --- GCS MODIFICATION END ---

        for filename in filenames:
            match = re.search(pattern, filename)
            # --- GCS MODIFICATION START ---
            dcp_metadata_probe = join([folder, filename, ".metadata"])
            safetensors_metadata_probe = join(
                [folder, filename, "model.safetensors.index.json"]
            )
            if match and isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and isfile(safetensors_metadata_probe):
            # --- GCS MODIFICATION END ---
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _ft_folder(self) -> str:
        # --- GCS MODIFICATION START ---
        return self.sep.join([self.folder, f"ft-replicat-{self.ft_replica_id}"])
        # --- GCS MODIFICATION END ---

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        # --- GCS MODIFICATION START ---
        return self.sep.join([folder, f"step-{step}"])
        # --- GCS MODIFICATION END ---

    def _ft_save(self, step: int) -> None:
        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = self.dcp_save(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            async_mode=AsyncMode.ASYNC,
            # --- FIX: Pass to_hf=False ---
            to_hf=False,
        )
        logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.")

    def _ft_load(self) -> None:
        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.dcp_load(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            # FT checkpoints are always DCP because FT checkpoint currently only save/load dataloader.
            from_hf=False,
            from_quantized=False,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.
        ... (rest of docstring) ...
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.
        ... (rest of docstring) ...
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load)

        if self.enable_ft_dataloader_checkpoints:
            states_to_load.pop(DATALOADER)

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # ... (rest of function) ...
        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif (
            self.async_mode == AsyncMode.ASYNC or self.enable_ft_dataloader_checkpoints
        ):
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _purge_stale_checkpoints(self):
        # --- GCS MODIFICATION START ---
        # Use fsspec-aware helpers for GCS paths
        isdir = self.fs.isdir if self.is_gcs else os.path.isdir
        listdir = (
            (lambda p: [f.split(self.sep)[-1] for f in self.fs.ls(p, detail=False)])
            if self.is_gcs
            else os.listdir
        )
        join = self.sep.join
        # --- GCS MODIFICATION END ---
        
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            # --- GCS MODIFICATION ---
            and isdir(self.folder)
            and (
                not self.enable_ft_dataloader_checkpoints
                or (self.ft_manager and self.ft_manager.participating_rank() == 0)
            )
        ):
            discovered_checkpoints = []
            try:
                # --- GCS MODIFICATION ---
                filenames = listdir(self.folder)
            except FileNotFoundError:
                filenames = [] # Folder might be empty

            for filename in filenames:
                match = re.search(r"step-(\d+)", filename)
                if match:
                    # --- GCS MODIFICATION ---
                    path = join([self.folder, filename])
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)