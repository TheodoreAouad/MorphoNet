"""Root logic for trainings."""

from pathlib import Path
import logging

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import mlflow.pytorch
import numpy as np

from misc.context import RunContext, Task
from misc.parser import parser, LOSSES
from misc.visualizer import VisualizerCallback
from datasets.base import DataModule
from models.base import VAL_LOSS, BaseNetwork
from operations.base import Operation
from operations.structuring_elements import StructuringElement
from tasks import output_managment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(levelname)s <%(module)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO filter size seems to be alsways int, but could be set to tuple
# TODO add param to change mlflow experiment
# TODO for lmorph, as inputs are in the range [1;2], the output should also be
# calculated from [1;2]: maybe harder to learn when starting in the wrong scale
# TODO normalisation des données
# TODO permettre de réduire la fréquence des print log (progress bar)
# TODO tester avec nombre de workers plus élever
# TODO check si les données sont bien en mémoire en permanance, peut-être la cause de la lenteur

mlflow.pytorch.autolog()
mlflow.set_tracking_uri(f"file://{Path(__file__).parents[2]}/mlruns")
if __name__ == "__main__":
    with Task("Parsing command line"):
        args = parser.parse_args()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run, RunContext(run, output_managment):
        with Task("Logging parameters"):
            mlflow.log_params(vars(args))

        with Task("Loading structuring element"):
            structuring_element_class = StructuringElement.select(
                name=args.structuring_element,
                filter_size=args.filter_size,
                precision=args.precision,
            )

            if structuring_element_class is None:
                logging.info("No matching structuring element found")
                structuring_element = np.empty(0)
            else:
                structuring_element = structuring_element_class()

        with Task("Loading operation"):
            operation = Operation.select(
                name=args.operation,
                structuring_element=structuring_element,
                percentage=args.percentage,
            )

        # TODO model should have available keys in args
        with Task("Loading model"):
            model = BaseNetwork.select(
                name=args.model,
                filter_size=args.filter_size,
                loss_function=LOSSES[args.loss](),
                # device=device,
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

        visualizer = VisualizerCallback(run, structuring_element, args.vis_freq)
        early_stop_callback = EarlyStopping(
            monitor=VAL_LOSS,
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min",  # TODO change for classif
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
