#include <Python.h>
#include <torch/script.h>
#include <cuda.h>

#include "LMorph.h"

static auto registry =
    torch::RegisterOperators()
        .op("lmorph::lmorph(Tensor input, Tensor filter, Tensor p) -> Tensor", &lmorph);
