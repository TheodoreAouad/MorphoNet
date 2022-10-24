"""Argument parser definiton."""

import argparse
from operations import OPS
from operations.structuring_elements import STRUCTURING_ELEMENTS

from misc.losses import LOSSES
from misc.utils import PRECISIONS_TORCH

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument("model", help="model to load")
parser.add_argument(
    "--loss", choices=LOSSES.keys(), help="loss to use", default="mse"
)

parser.add_argument(
    "--op",
    default=None,
    choices=OPS,
    help="operation to perform",
    dest="operation",
)
parser.add_argument(
    "--sel",
    choices=STRUCTURING_ELEMENTS,
    help="structuring element to use",
    default="",
    dest="structuring_element",
)
parser.add_argument(
    "--filter_size",
    type=int,
    default=9,
    help="size of the filter to apply and learn",
)
parser.add_argument(
    "--percentage", type=int, default=0, help="percentage of noise"
)

parser.add_argument(
    "--epochs", type=int, default=100, help="number of epochs to train for"
)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="number of epochs to continue training for with no improvement to validation loss",
)

parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument(
    "--validation_split", type=float, default=0.1, help="validation split"
)

parser.add_argument(
    "--precision", choices=PRECISIONS_TORCH.keys(), default="f64"
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="which GPU to restrict the training session to",
)
parser.add_argument(
    "--vis_freq",
    type=int,
    default=32,
    help="how often to save layers outputs and weights (in batches)",
)
parser.add_argument(
    "--experiment",
    type=str,
    default="Default",
    help="Name of the MLflow experiment",
)

subparsers = parser.add_subparsers(help="Dataset to train on", dest="dataset")

# TODO make a loop creating subparsers (looping on datasets.DATASETS)

mnist_parser = subparsers.add_parser("mnist", help="Train on MNIST")
mnist_parser.add_argument("dataset_path", help="dataset to train on")

biwtoh_parser = subparsers.add_parser("biwtoh", help="Train on BiWToH")
biwtoh_parser.add_argument("dataset_path", help="dataset to train on")

gwtoh_parser = subparsers.add_parser("gwtoh", help="Train on BiWToH")
gwtoh_parser.add_argument("dataset_path", help="dataset to train on")

fashion_mnist_parser = subparsers.add_parser(
    "fashion_mnist", help="Train on FashionMNIST"
)
fashion_mnist_parser.add_argument("dataset_path", help="dataset to train on")

sidd_parser = subparsers.add_parser("sidd", help="Train on sidd")
sidd_parser.add_argument("dataset_path", nargs=1, help="dataset to train on")
sidd_parser.add_argument(
    "--patch_size",
    type=int,
    default=64,
    help="width and height of the image patches",
)
sidd_parser.add_argument(
    "--smartphone_codes",
    default=None,
    help="which smartphone codes to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--iso_levels",
    default=None,
    help="which iso levels to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--shutter_speeds",
    default=None,
    help="which shutter speeds to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--illuminants",
    default=None,
    help="which illuminants to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--ibcs",
    default=None,
    help="which illuminant brightness codes to use from the dataset (comma-separated)",
)
