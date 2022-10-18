"""Root logic for trainings."""

from lib2to3.pgen2.token import OP
from misc.visualizer import VisualizerCallback
from misc.context import OutputManagment

output_managment = OutputManagment()
output_managment.set()

import logging
import mlflow.pytorch
import os

# Same device ordering as `nvidia-smi`
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import torch

from misc.context import RunContext, Task

from misc.parser import parser, LOSSES, PRECISIONS_TORCH
from datasets.base import DataModule
from models.base import VAL_LOSS
from operations.base import Operation
from operations.structuring_elements import StructuringElement
from models.base import BaseNetwork

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(levelname)s <%(module)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO filter size seems to be alsways int, but could be set to tuple
# TODO add param to change mlflow experiment

mlflow.pytorch.autolog()
if __name__ == "__main__":
    with mlflow.start_run() as run, RunContext(run, output_managment):
        with Task("Parsing command line"):
            args = parser.parse_args()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with Task("Logging parameters"):
            mlflow.log_params(vars(args))

        with Task("Loading structuring element"):
            structuring_element = StructuringElement.select(
                name=args.structuring_element,
                filter_size=args.filter_size,
                precision=args.precision
            )
            
            if structuring_element == None:
                logging.info("No matching structuring element found")

        with Task("Loading operation"):
            operation = Operation.select(
                name=args.operation,
                structuring_element=structuring_element(),
                percentage=args.percentage
            )

        # TODO model should have available keys in args
        with Task("Loading model"):
            model = BaseNetwork.select(
                name=args.model,
                filter_size=args.filter_size,
                loss_function=LOSSES[args.loss](),
                #device=device,
            )

        # Accuracy of best val loss comparison
        # TODO check dtype for the modle (should be same as arg)

        with Task("Loading dataset"):
            data_module = DataModule.select(
                name=args.dataset,
                dataset_path=args.dataset_path,
                batch_size=args.batch_size,
                precision=args.precision,
                operation=operation,
            )

        visualizer = VisualizerCallback(run, structuring_element(), args.vis_freq)
        early_stop_callback = EarlyStopping(
            monitor=VAL_LOSS,
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min", # TODO change for classif
        )

        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[early_stop_callback, visualizer],
            log_every_n_steps=1,
            accelerator="gpu",
            devices=[args.gpu],
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
        )
