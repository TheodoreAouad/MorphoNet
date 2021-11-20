import pathlib
import json
from functools import partial

import numpy as np
import h5py
import torch


def ensure_dir(path, parent=False):
    path = pathlib.Path(path)
    path = path.parent if parent else path
    path.mkdir(parents=True, exist_ok=True)


class ModuleVisualizer:
    def __init__(
        self, inputs, targets, sel, module, out_dir, filter_children=None, freq=16,
        batch_size=0, max_epochs=0, sel_name="", patience=0, module_name="",
        loss="", dataset="", percentage=None
    ):
        self.inputs = inputs
        self.targets = targets
        self.sel = sel
        self.module = module
        self.module_name = module_name
        if filter_children is not None:
            self.children = list(
                filter(
                    lambda mod: mod[0] in filter_children, self.module.named_children(),
                )
            )
        else:
            self.children = list(self.module.named_children())
        self.out_dir = out_dir
        self.freq = freq
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss= loss
        self.dataset = dataset
        self.sel_name = sel_name
        self.patience = patience 
        self.current_batch = 0
        self.saved_batch = 0
        self.fn = None
        self.current_epoch = 0
        self.percentage = percentage

        ensure_dir(f"{self.out_dir}/batches")

        with h5py.File(f"{self.out_dir}/meta.h5", "w") as f:
            f.create_dataset("inputs", data=self.inputs.cpu().numpy())
            f.create_dataset("targets", data=self.targets.cpu().numpy())
            if self.sel_name != None:
                f.create_dataset("sel", data=self.sel)
                f.create_dataset("sel_name", data=self.sel_name)
                f.create_dataset("sel_size", data=self.sel.shape[0])
            else:
                f.create_dataset("percentage", data=self.percentage)
            f.create_dataset("batch_size", data=self.batch_size)
            f.create_dataset("max_epochs", data=self.max_epochs)
            f.create_dataset("vis_freq", data=self.freq)
            f.create_dataset("model", data=self.module_name)
            f.create_dataset("patience", data=self.patience)
            f.create_dataset("loss", data=self.loss)
            f.create_dataset("dataset", data=self.dataset)

        self._save({})
        self.current_batch += 1

    def _save(self, logs):
        outputs = {}
        handles = []

        def add_output(name, module, input, output):
            outputs[name] = output.cpu().detach().numpy()

        for name, child in self.children:
            handles.append(child.register_forward_hook(partial(add_output, name)))

        with torch.no_grad():
            self.module(self.inputs)

        for handle in handles:
            handle.remove()

        logs = {
            **logs,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
        }

        with h5py.File(f"{self.out_dir}/batches/{self.saved_batch:06}.h5", "w") as f:
            logs_group = f.create_group("logs")

            for key, value in logs.items():
                logs_group.create_dataset(key, data=value)

            layers_group = f.create_group("layers")

            for idx, (name, child) in enumerate(self.children):
                layer_group = layers_group.create_group(f"{idx:03}_{name}")
                weights_group = layer_group.create_group("weights")

                for param_name, param in child.named_parameters():
                    weights_group.create_dataset(
                        param_name, data=param.data.cpu().numpy(),
                    )

                for buffer_name, buffer in child.named_buffers():
                    weights_group.create_dataset(
                        buffer_name, data=buffer.data.cpu().numpy(),
                    )

                layer_group.create_dataset("output", data=outputs[name])

            for param_name, param in self.module.named_parameters():
                if '.' not in param_name:
                    layers_group.create_dataset(
                        param_name, data=param.data.cpu().numpy(),
                    )

        self.saved_batch += 1

    def step_epoch(self):
        self.current_epoch += 1

    def step_batch(self, logs):
        if self.current_batch % self.freq == 0:
            self._save(logs)

        self.current_batch += 1

    def finish(self, logs):
        with h5py.File(f"{self.out_dir}/meta.h5", "a") as f:
            for key, value in logs.items():
                f.create_dataset(key, data=value)



