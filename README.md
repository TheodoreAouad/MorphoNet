# MorphoNet

> __Author:__ Romain HERMARY <<romain.hermary@lrde.epita.fr>>
>
> __Linked Articles:__
> - [Learning Grayscale Mathematical Morphology with Smooth Morphological Layers](https://link.springer.com/article/10.1007/s10851-022-01091-1)
> - [Going beyond p-convolutions to learn grayscale morphological operators](https://arxiv.org/abs/2102.10038)

## Description

This repository is a complete reimplementation of the code used to write the linked articles. I chose to recode from scratch to introduce a cleaner object-oriented implementation, while also adding the use of management libraries such as [MLFlow](https://www.mlflow.org/), [PyTorch Lightning](https://pytorchlightning.ai/), TensorFlow's [TensorBoard](https://www.tensorflow.org/tensorboard) and remove any direct dependency to [CUDA](https://developer.nvidia.com/cuda-toolkit) using [PyTorch](https://pytorch.org/) framework for everything.

I use [Black](https://black.readthedocs.io/en/stable/), [Mypy](https://mypy.readthedocs.io/en/stable/index.html) and [Pylint](https://pylint.pycqa.org/en/latest/) as linters (`pyproject.toml` is the config file).

## Example

A similar script example can be found in the `scripts/run.sh` file.
Other scripts are also in the same folder, examples of how I launch the benchmarks.

```bash
$ python -m tasks.train\
            --filter_size 7\
            --epochs 1000\
            --gpu 0\
            --patience 10\
            --vis_freq 125\
            --op dilation\
            --sel complex\
            --loss mse\
            --experiment baseline\
            smorphnet\
            mnist ./data/mnist
```

I personally use the code in a Python [virtual environment](https://docs.python.org/3/library/venv.html).

> Note: If you train for a denoising task, the `--sel` argument is automatically ignored; on the other hand, if you train for a morphological operation, the `--percentage` is automatically ignored.

## Other Information

Bunch of things are not yet reimplemented or perfect, but you can already find here a lot of the papers' basis. I will also make sure more tests are written.

I have already worked on the denoising and classification subjects using the morphological layers, as well as trying new structures able to learn other morphological operations such as the white top-hat. I will reimplement those here properly in the future for the project to go further in a very stable manner.

There is a Notebook in the `analysis/` folder where you can visualize the learned structuring elements. You can also have a quick pick on the results if you start __MLFlow__ and/or __TensorBoard__ servers (saves should be under the `mlruns/` and `lightning_logs/` folders respectively).

```bash
$ mlflow server
$ tensorboard --log_dir lightning_logs
```

<br><br><br>

<p style="text-align: center;">
If any questions or remarks, please feel free to contact me!
</p>
