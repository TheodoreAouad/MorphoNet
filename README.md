# morphonet

- `training/` Training, models, and other utilities.
- `lmorph/` LehmerMorph CUDA/torch implementation.
- `smorph/` SmoothMorph CUDA/torch implementation.

## Usage

First, install both the `smorph` and `lmorph` packages. Go into the respective directories and execute `install.sh`. You'll need a C/C++ compiler. See also https://github.com/pytorch/extension-cpp.

Then, to train one of the models in `training/models/`:

```sh
# Prints the help
python3.7 -m task.train --help

# Example training:
python3.7 -m task.train --filter_size 7 --out_dir /lrde-work/node9/$USER/vis --epochs 1000 --vis_freq 512 --batch_size 32 --patience 50 models.smorphnet_double mse closing complex mnist /lrde-work/node9/$USER/mnist
```

The example training above trains with the following parameters:
- on the MNIST dataset located over at `/lrde-work/node9/$USER/mnist`
- learning a filter of size 7x7
- for the `smorphnet_double` model (see training/models)
- on the closing operation on the "complex" structuring element (see training/preprocessing/sel.py)
- for 1000 epochs, with a batch size of 32, exporting visualisation data every 512 batches (see the morphovis repo)
- using the MSE loss

There's also support for the SSIM loss and the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) dataset, which you'll need to have downloaded somewhere. See also the options to the training script.