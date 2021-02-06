#include <Python.h>
#include <torch/script.h>
#include <cuda.h>

#include "SMorph.h"

static auto registry =
    torch::RegisterOperators()
        .op("smorph::smorph(Tensor input, Tensor filter, Tensor alpha) -> Tensor", &smorph);
