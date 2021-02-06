#pragma once

#include <torch/extension.h>

#include "cuda/SMorph_cuda.h"

std::tuple<at::Tensor, at::Tensor> SMorph_forward(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &alpha)
{
  return SMorph_forward_cuda(input, filter, alpha);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> SMorph_backward(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    const at::Tensor &lower,
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &alpha)
{
  return SMorph_backward_cuda(
      grad_output,
      output,
      lower,
      input,
      filter,
      alpha);
}

class SMorphFunction
    : public torch::autograd::Function<SMorphFunction>
{
public:
  static variable_list forward(
      AutogradContext *ctx,
      Variable input,
      Variable filter,
      Variable alpha)
  {
    auto result = SMorph_forward(
        input,
        filter,
        alpha);
    auto output = std::get<0>(result);
    auto lower = std::get<1>(result);
    ctx->save_for_backward({output, lower, input, filter, alpha});
    return {output};
  }

  static variable_list backward(
      AutogradContext *ctx,
      variable_list grad_output)
  {
    auto saved = ctx->get_saved_variables();
    auto output = saved[0];
    auto lower = saved[1];
    auto input = saved[2];
    auto filter = saved[3];
    auto alpha = saved[4];
    auto result = SMorph_backward(
        grad_output[0],
        output,
        lower,
        input,
        filter,
        alpha);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

at::Tensor smorph(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &alpha)
{
  auto result = SMorphFunction::apply(
      input, filter, alpha);
  return result[0];
}
