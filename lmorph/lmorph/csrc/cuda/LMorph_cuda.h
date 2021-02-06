#pragma once

#include <torch/extension.h>

#include "ValidDim.h"

std::tuple<at::Tensor, at::Tensor> LMorph_forward_cuda(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p);

std::tuple<at::Tensor, at::Tensor, at::Tensor> LMorph_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    const at::Tensor &lower,
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p);
