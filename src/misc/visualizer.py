"""Module allowing the visualization of the training details."""

from typing import Any, Optional, Union, Type
from contextlib import redirect_stdout
import pickle
import os
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.utilities.model_summary import (
    DeepSpeedSummary,
    ModelSummary,
    summarize,
)
from pytorch_lightning.utilities.model_summary.model_summary import (
    _format_summary_table,
)

from mlflow.utils.file_utils import local_file_uri_to_path  # type: ignore
from mlflow import ActiveRun

from operations.structuring_elements import StructuringElement

BASE_DIR = "outputs"


class VisualizerCallback(
    Callback
):  # pylint: disable=too-many-instance-attributes
    """Callback in charge of logging during training."""

    def __init__(
        self,
        run: ActiveRun,
        structuring_element: StructuringElement,
        frequency: int = 16,
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.current_batch = 0
        self.saved_batch = 0
        self._max_depth = 1
        try:
            self.structuring_element: Optional[
                np.ndarray
            ] = structuring_element()
        except NotImplementedError:
            self.structuring_element = None

        self.artifact_path = local_file_uri_to_path(run.info.artifact_uri)
        self.artifact_format_pickle = (
            f"{self.artifact_path}/{BASE_DIR}/{{}}.pickle".format
        )
        self.artifact_format_ckpt = (
            f"{self.artifact_path}/{BASE_DIR}/{{}}.ckpt".format
        )

        self.base_path = f"{self.artifact_path}/{BASE_DIR}"
        os.mkdir(self.base_path)

        self.weigths_plots_path = f"{self.base_path}/weights_plots/"
        os.mkdir(self.weigths_plots_path)

        self.inputs: torch.Tensor

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        self.inputs = (
            trainer.datamodule.sample[0]  # type: ignore
            .detach()
            .clone()
            .to(trainer.strategy.root_device)
        )
        targets = trainer.datamodule.sample[1].detach().clone().to("cpu")  # type: ignore

        # TODO finish implem meta file
        # TODO image grid print of inputs/targets on TB
        with open(self.artifact_format_pickle("meta"), "wb") as meta_file:
            pickle.dump(
                {
                    "inputs": self.inputs.cpu(),
                    "targets": targets,
                    "structuring_element": self.structuring_element,
                },
                meta_file,
            )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

        if self.current_batch % self.frequency == 0:
            trainer.save_checkpoint(
                self.artifact_format_ckpt(f"{self.saved_batch:06}")
            )

            self.saved_batch += 1

        self.current_batch += 1

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        For each compatible layer, plot the weights for direct visualization.
        """
        index, handles = 0, []

        def add_hook(module_: torch.nn.Module) -> None:
            def forward_hook(  # pylint: disable=unused-argument
                module__: torch.nn.Module,
                module_input: torch.Tensor,
                module_output: torch.Tensor,
            ) -> None:
                nonlocal index

                if hasattr(module_, "plot"):
                    plot_method = getattr(module_, "plot")
                    try:
                        path = f"{self.weigths_plots_path}/{index}_{type(module_).__name__.lower()}.png"
                        plot_method(path=path, target=self.structuring_element)
                    except NotImplementedError:
                        pass

                index += 1

            nonlocal handles
            handles.append(module_.register_forward_hook(forward_hook))

        pl_module.apply(add_hook)

        with torch.no_grad():
            pl_module.predict_step(self.inputs, -1)

        for handle in handles:
            handle.remove()

    @staticmethod
    def plot_saved_model(
        model_class: Type, ckpt_path: str, log_dir: str, inputs: torch.Tensor
    ) -> None:
        """
        For each compatible layer, plot the weights of the given model
        checkpoint for direct visualization.
        """
        index, handles = 0, []
        pl_module = model_class.load_from_checkpoint(ckpt_path)

        os.mkdir(log_dir)

        def add_hook(module_: torch.nn.Module) -> None:
            def forward_hook(  # pylint: disable=unused-argument
                module__: torch.nn.Module,
                module_input: torch.Tensor,
                module_output: torch.Tensor,
            ) -> None:
                nonlocal index

                if hasattr(module_, "plot"):
                    plot_method = getattr(module_, "plot")
                    try:
                        path = f"{log_dir}/{index}_{type(module_).__name__.lower()}.png"
                        plot_method(path=path)
                    except NotImplementedError:
                        print("skipping", module_.__class__.__name__)

                index += 1

            nonlocal handles
            handles.append(module_.register_forward_hook(forward_hook))

        pl_module.apply(add_hook)

        with torch.no_grad():
            pl_module.predict_step(inputs, -1)

        for handle in handles:
            handle.remove()

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Taken from pytorch_lightning.callbacks.model_summary.py
        Print a summary of the model to a `model_summary.txt` file
        """
        if not self._max_depth or not trainer.is_global_zero:
            return

        model_summary: Union[DeepSpeedSummary, ModelSummary]
        if (
            isinstance(trainer.strategy, DeepSpeedStrategy)
            and trainer.strategy.zero_stage_3
        ):
            model_summary = DeepSpeedSummary(
                pl_module, max_depth=self._max_depth
            )
        else:
            model_summary = summarize(pl_module, max_depth=self._max_depth)

        summary_data = (
            model_summary._get_summary_data()  # pylint: disable=protected-access
        )
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size

        summary_table = _format_summary_table(
            total_parameters, trainable_parameters, model_size, *summary_data
        )
        with open(
            f"{self.artifact_path}/model_summary.txt", "w", encoding="utf-8"
        ) as summary_file:
            with redirect_stdout(summary_file):
                print(summary_table)
