"""Utility functions."""

from torch.utils.data import DataLoader
import torch
import numpy as np


PRECISIONS_TORCH = {"f32": torch.float32, "f64": torch.float64}
PRECISIONS_NP = {"f32": "float32", "f64": "float64"}

# TODO do something with this file
# pylint: disable=all
import torchvision
import torch


def fit_NCHW(tensor: torch.Tensor, include_c: bool = False) -> torch.Tensor:
    return torchvision.transforms.Lambda(lambda x: _fit_NCHW(x, include_c))(
        tensor
    )


def _fit_NCHW(tensor: torch.Tensor, include_c: bool = False) -> torch.Tensor:
    if len(tensor.size()) == 4:
        return tensor  # Already in NCHW format

    if len(tensor.size()) == 2:  # Only H and W dimensions
        return tensor[None, None, :, :]

    if include_c:  # 3D data (C, H, W) with only one sample
        return tensor[None, :, :, :]

    return tensor[:, None, :, :]  # 2D data with N samples (N, C, W)


def split_arg(value, mapper=lambda a: a):  # type: ignore
    if value is None:
        return None
    return [mapper(v) for v in value.split(",")]


def get_data(train_ds, valid_ds, bs):  # type: ignore
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def pad_inputs(x, model_name, filter_padding, pad_value=0):  # type: ignore
    if "double" in model_name:
        filter_padding *= 2
    elif "five" in model_name:
        filter_padding *= 5
    elif "four" in model_name:
        filter_padding *= 4

    padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (filter_padding, filter_padding),
            (filter_padding, filter_padding),
        ),
        mode="constant",
        constant_values=(pad_value,),
    )

    return padded


def make_patches(data_shape, patch_shape):  # type: ignore
    patch_shape = np.array(patch_shape)
    data_shape = np.array(data_shape)
    n_patches = np.ceil(data_shape / patch_shape)
    x, y = np.mgrid[
        0 : data_shape[0] - patch_shape[0] : n_patches[0] * 1j,
        0 : data_shape[1] - patch_shape[1] : n_patches[1] * 1j,
    ]
    translations = np.round(np.array([x, y]).T.reshape((-1, 2))).astype(int)
    return [
        (slice(t[0], t[0] + patch_shape[0]), slice(t[1], t[1] + patch_shape[1]))
        for t in translations
    ]


def reconstruct_patches(data_patches, patches, data_shape):  # type: ignore
    data = np.zeros(data_shape, dtype=data_patches.dtype)
    count = np.zeros(data_shape, dtype=data_patches.dtype)
    for data_patch, patch in zip(data_patches, patches):
        data[patch] += data_patch
        count[patch] += 1
    data = data / count
    return data


import random


def _generate_string(sep, integer_scale):  # type: ignore

    predicate = random.choice(_GENERATOR_PREDICATES).lower()
    noun = random.choice(_GENERATOR_NOUNS).lower()
    num = random.randint(0, 10**integer_scale)
    return f"{predicate}{sep}{noun}{sep}{num}"


def _generate_random_name(sep="-", integer_scale=3, max_length=20):  # type: ignore
    """Helper function for generating a random predicate, noun, and integer combination

    :param sep: String seperator for word spacing
    :param integer_scale: dictates the maximum scale range for random integer sampling (power of 10)
    :param max_length: maximum allowable string length

    :return: A random string phrase comprised of a predicate, noun, and random integer
    """
    name = None
    for _ in range(10):
        name = _generate_string(sep, integer_scale)
        if len(name) <= max_length:
            return name
    # If the combined length isn't below the threshold after 10 iterations, truncate it.
    return name[:max_length]


_GENERATOR_NOUNS = [
    "ant",
    "ape",
    "asp",
    "auk",
    "bass",
    "bat",
    "bear",
    "bee",
    "bird",
    "boar",
    "bug",
    "calf",
    "carp",
    "cat",
    "chick",
    "chimp",
    "cod",
    "colt",
    "conch",
    "cow",
    "crab",
    "crane",
    "croc",
    "crow",
    "cub",
    "deer",
    "doe",
    "dog",
    "dolphin",
    "donkey",
    "dove",
    "duck",
    "eel",
    "elk",
    "fawn",
    "finch",
    "fish",
    "flea",
    "fly",
    "foal",
    "fowl",
    "fox",
    "frog",
    "gnat",
    "gnu",
    "goat",
    "goose",
    "grouse",
    "grub",
    "gull",
    "hare",
    "hawk",
    "hen",
    "hog",
    "horse",
    "hound",
    "jay",
    "kit",
    "kite",
    "koi",
    "lamb",
    "lark",
    "loon",
    "lynx",
    "mare",
    "midge",
    "mink",
    "mole",
    "moose",
    "moth",
    "mouse",
    "mule",
    "newt",
    "owl",
    "ox",
    "panda",
    "penguin",
    "perch",
    "pig",
    "pug",
    "quail",
    "ram",
    "rat",
    "ray",
    "robin",
    "roo",
    "rook",
    "seal",
    "shad",
    "shark",
    "sheep",
    "shoat",
    "shrew",
    "shrike",
    "shrimp",
    "skink",
    "skunk",
    "sloth",
    "slug",
    "smelt",
    "snail",
    "snake",
    "snipe",
    "sow",
    "sponge",
    "squid",
    "squirrel",
    "stag",
    "steed",
    "stoat",
    "stork",
    "swan",
    "tern",
    "toad",
    "trout",
    "turtle",
    "vole",
    "wasp",
    "whale",
    "wolf",
    "worm",
    "wren",
    "yak",
    "zebra",
]

_GENERATOR_PREDICATES = [
    "abundant",
    "able",
    "abrasive",
    "adorable",
    "adaptable",
    "adventurous",
    "aged",
    "agreeable",
    "ambitious",
    "amazing",
    "amusing",
    "angry",
    "auspicious",
    "awesome",
    "bald",
    "beautiful",
    "bemused",
    "bedecked",
    "big",
    "bittersweet",
    "blushing",
    "bold",
    "bouncy",
    "brawny",
    "bright",
    "burly",
    "bustling",
    "calm",
    "capable",
    "carefree",
    "capricious",
    "caring",
    "casual",
    "charming",
    "chill",
    "classy",
    "clean",
    "clumsy",
    "colorful",
    "crawling",
    "dapper",
    "debonair",
    "dashing",
    "defiant",
    "delicate",
    "delightful",
    "dazzling",
    "efficient",
    "enchanting",
    "entertaining",
    "enthused",
    "exultant",
    "fearless",
    "flawless",
    "fortunate",
    "fun",
    "funny",
    "gaudy",
    "gentle",
    "gifted",
    "glamorous",
    "grandiose",
    "gregarious",
    "handsome",
    "hilarious",
    "honorable",
    "illustrious",
    "incongruous",
    "indecisive",
    "industrious",
    "intelligent",
    "inquisitive",
    "intrigued",
    "invincible",
    "judicious",
    "kindly",
    "languid",
    "learned",
    "legendary",
    "likeable",
    "loud",
    "luminous",
    "luxuriant",
    "lyrical",
    "magnificent",
    "marvelous",
    "masked",
    "melodic",
    "merciful",
    "mercurial",
    "monumental",
    "mysterious",
    "nebulous",
    "nervous",
    "nimble",
    "nosy",
    "omniscient",
    "orderly",
    "overjoyed",
    "peaceful",
    "painted",
    "persistent",
    "placid",
    "polite",
    "popular",
    "powerful",
    "puzzled",
    "rambunctious",
    "rare",
    "rebellious",
    "respected",
    "resilient",
    "righteous",
    "receptive",
    "redolent",
    "resilient",
    "rogue",
    "rumbling",
    "salty",
    "sassy",
    "secretive",
    "selective",
    "sedate",
    "serious",
    "shivering",
    "skillful",
    "sincere",
    "skittish",
    "silent",
    "smiling",
    "sneaky",
    "sophisticated",
    "spiffy",
    "stately",
    "suave",
    "stylish",
    "tasteful",
    "thoughtful",
    "thundering",
    "traveling",
    "treasured",
    "trusting",
    "unequaled",
    "upset",
    "unique",
    "unleashed",
    "useful",
    "upbeat",
    "unruly",
    "valuable",
    "vaunted",
    "victorious",
    "welcoming",
    "whimsical",
    "wistful",
    "wise",
    "worried",
    "youthful",
    "zealous",
]
