"""Module allowing the visualization of the training details."""

from typing import Any, Optional
import pickle
import os
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from mlflow.utils.file_utils import local_file_uri_to_path  # type: ignore
from mlflow import ActiveRun

BASE_DIR = "outputs"

class VisualizerCallback(Callback):
    """Callback in charge of logging during training."""

    def __init__(
        self,
        run: ActiveRun,
        structuring_element: np.ndarray,
        frequency: int = 16,
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.current_batch = 0
        self.saved_batch = 0
        self.structuring_element = structuring_element

        self.artifact_path = local_file_uri_to_path(run.info.artifact_uri)
        self.artifact_format = (
            f"{self.artifact_path}/{BASE_DIR}/{{}}.pickle".format
        )

        os.mkdir(f"{self.artifact_path}/{BASE_DIR}")

    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        self.inputs = trainer.datamodule.sample[0].detach().clone().to(
            trainer.strategy.root_device
        )
        targets = trainer.datamodule.sample[1].detach().clone().to("cpu")
        
        # TODO finish implem meta file
        with open(self.artifact_format("meta"), "wb") as meta_file:
            pickle.dump(
                {
                    "inputs": self.inputs.cpu(),
                    "targets": targets,
                    "structuring_element": self.structuring_element,
                },
                meta_file,
            )

    def _save(self, module: pl.LightningModule) -> None:
        outputs, weights, handles = [], [], []
        index = 0

        def add_hook(module_: torch.nn.Module) -> None:
            def forward_hook(  # pylint: disable=unused-argument
                module__: torch.nn.Module,
                module_input: torch.Tensor,
                module_output: torch.Tensor,
            ) -> None:
                nonlocal index
                nonlocal outputs
                nonlocal weights
                # TODO only save output, index and name on init (meta)
                # TODO save les outputs de tous les layers seulement si debug max
                # (i.e créer un mode benchmarck pour save seulement l'output du net)

                outputs.append(
                    (index, type(module__).__name__, module_output.cpu())
                )

                module_data = {}

                for param_name, param in module__.named_parameters():
                    module_data[param_name] = param.data.cpu()

                for buffer_name, buffer in module__.named_buffers():
                    module_data[buffer_name] = buffer.data.cpu()

                weights.append((index, type(module__).__name__, module_data))
                index += 1

            nonlocal handles
            handles.append(module_.register_forward_hook(forward_hook))

        module.apply(add_hook)

        with torch.no_grad():
            module.predict_step(self.inputs, -1)

        with open(
            self.artifact_format(f"{self.saved_batch:06}"), "wb"
        ) as dump_file:
            pickle.dump(
                {
                    "network_outputs": outputs[:-1],
                    "layers_weights": weights[:-1],
                },
                dump_file,
            )

        for handle in handles:
            handle.remove()

        self.saved_batch += 1

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

        if self.current_batch % self.frequency == 0:
            self._save(pl_module)

        self.current_batch += 1
