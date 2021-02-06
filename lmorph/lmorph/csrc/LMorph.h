#pragma once

#include <torch/extension.h>

#include "cuda/LMorph_cuda.h"

std::tuple<at::Tensor, at::Tensor> LMorph_forward(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p)
{
  return LMorph_forward_cuda(input, filter, p);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LMorph_backward(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    const at::Tensor &lower,
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p)
{
  return LMorph_backward_cuda(
      grad_output,
      output,
      lower,
      input,
      filter,
      p);
}

class LMorphFunction
    : public torch::autograd::Function<LMorphFunction>
{
public:
  static variable_list forward(
      AutogradContext *ctx,
      Variable input,
      Variable filter,
      Variable p)
  {
    auto result = LMorph_forward(
        input,
        filter,
        p);
    auto output = std::get<0>(result);
    auto lower = std::get<1>(result);
    ctx->save_for_backward({output, lower, input, filter, p});
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
    auto p = saved[4];
    auto result = LMorph_backward(
        grad_output[0],
        output,
        lower,
        input,
        filter,
        p);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

at::Tensor lmorph(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p)
{
  auto result = LMorphFunction::apply(
      input, filter, p);
  return result[0];
}
