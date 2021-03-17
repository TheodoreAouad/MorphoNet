#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "ValidDim.h"
#include "vscode_cuda.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

namespace
{

template <typename scalar_t>
__global__ void LMorphForwardCUDA(
    const size_t output_size,
    const ValidDim dim,
    const scalar_t *input,
    const scalar_t *filter,
    const scalar_t *p,
    scalar_t *lower,
    scalar_t *out)
{
  CUDA_1D_KERNEL_LOOP(index, output_size)
  {
    const size_t output_col = index % dim.output_cols;
    const size_t output_row = (index / dim.output_cols) % dim.output_rows;
    const size_t output_channel = (index / dim.output_cols / dim.output_rows) % dim.output_channels;
    const size_t batch = index / dim.output_cols / dim.output_rows / dim.output_channels;

    const size_t input_batch_offset = batch * dim.input_batch_stride;
    const size_t filter_output_offset = output_channel * dim.filter_output_stride;
    const size_t p_output_offset = output_channel * dim.input_channels;

    scalar_t upper_s = 0.0;
    scalar_t lower_s = 0.0;
    for (size_t input_channel = 0; input_channel < dim.input_channels; ++input_channel)
    {
      const size_t input_channel_offset = input_batch_offset + input_channel * dim.input_channel_stride;
      const size_t filter_channel_offset = filter_output_offset + input_channel * dim.filter_input_stride;
      scalar_t p_s = p[p_output_offset + input_channel];
      for (size_t filter_row = 0; filter_row < dim.filter_rows; ++filter_row)
      {
        const size_t input_row_offset = input_channel_offset + (output_row + filter_row) * dim.input_row_stride;
        const size_t filter_row_offset = filter_channel_offset + filter_row * dim.filter_row_stride;
        for (size_t filter_col = 0; filter_col < dim.filter_cols; ++filter_col)
        {
          scalar_t neighbor = input[input_row_offset + output_col + filter_col];
          scalar_t neighbor_filter = filter[filter_row_offset + filter_col];
          scalar_t sum = neighbor + neighbor_filter;
          scalar_t res_lower = pow(sum, p_s);
          scalar_t res_upper = res_lower * sum;
          lower_s += res_lower;
          upper_s += res_upper;
        }
      }
    }
    lower[index] = lower_s;
    out[index] = upper_s / lower_s;
  }
}

template <typename scalar_t>
__global__ void LMorphBackwardCUDA(
    const size_t output_size,
    const ValidDim dim,
    const scalar_t *grad_output,
    const scalar_t *output,
    const scalar_t *lower,
    const scalar_t *input,
    const scalar_t *filter,
    const scalar_t *p,
    scalar_t *grad_input,
    scalar_t *grad_filter,
    scalar_t *grad_p)
{
  CUDA_1D_KERNEL_LOOP(index, output_size)
  {
    const size_t output_col = index % dim.output_cols;
    const size_t output_row = (index / dim.output_cols) % dim.output_rows;
    const size_t output_channel = (index / dim.output_cols / dim.output_rows) % dim.output_channels;
    const size_t batch = index / dim.output_cols / dim.output_rows / dim.output_channels;

    scalar_t lower_s = lower[index];
    scalar_t output_s = output[index];
    scalar_t grad_output_s = grad_output[index];

    const size_t input_batch_offset = batch * dim.input_batch_stride;
    const size_t filter_output_offset = output_channel * dim.filter_output_stride;
    const size_t p_output_offset = output_channel * dim.input_channels;

    for (size_t input_channel = 0; input_channel < dim.input_channels; ++input_channel)
    {
      const size_t input_channel_offset = input_batch_offset + input_channel * dim.input_channel_stride;
      const size_t filter_input_offset = filter_output_offset + input_channel * dim.filter_input_stride;
      const size_t p_input_offset = p_output_offset + input_channel;
      scalar_t p_s = p[p_input_offset];
      scalar_t grad_p_input_s = 0.0;

      for (size_t filter_row = 0; filter_row < dim.filter_rows; ++filter_row)
      {
        const size_t input_row_offset = input_channel_offset + (output_row + filter_row) * dim.input_row_stride;
        const size_t filter_row_offset = filter_input_offset + filter_row * dim.filter_row_stride;
        for (size_t filter_col = 0; filter_col < dim.filter_cols; ++filter_col)
        {
          const size_t input_col_offset = input_row_offset + output_col + filter_col;
          const size_t filter_col_offset = filter_row_offset + filter_col;
          scalar_t sum = input[input_col_offset] + filter[filter_col_offset];
          scalar_t sum_p_m1 = pow(sum, p_s - 1.0);
          scalar_t sum_p = sum_p_m1 * sum;
          scalar_t sum_p_p1 = sum_p * sum;
          scalar_t sum_log = log(sum);

          // Be careful with the order of operations here: `upper`, `lower` and `sum` can get very big.
          // As such, they should never be multiplied together directly, lest `inf` values appear in
          // our gradient. What we do here is divide big numbers together before multiplying them again.
          scalar_t grad_input_filter_s = (p_s + 1.0) * (sum_p / lower_s) - p_s * (sum_p_m1 / lower_s) * output_s;
          grad_input_filter_s *= grad_output_s;

          scalar_t grad_p_s = sum_log * (sum_p_p1 / lower_s) - sum_log * (sum_p / lower_s) * output_s;
          grad_p_input_s += grad_p_s;

          // Atomic add: multiple kernels might be trying to concurrently operate on the same memory location,
          // which would cause operations to be dropped non-deterministically.
          atomicAdd(grad_input + input_col_offset, grad_input_filter_s);
          atomicAdd(grad_filter + filter_col_offset, grad_input_filter_s);
        }
      }

      atomicAdd(grad_p + p_input_offset, grad_p_input_s * grad_output_s);
    }
  }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> LMorph_forward_cuda(
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p)
{
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(filter.device().is_cuda(), "filter must be a CUDA tensor");
  AT_ASSERTM(p.device().is_cuda(), "p must be a CUDA tensor");

  auto dim = ValidDim(input.sizes(), filter.sizes(), p.sizes());

  at::TensorArg input_t{input, "input", 1}, filter_t{filter, "filter", 2}, p_t{p, "p", 3};

  at::CheckedFrom c = "LMorph_forward_cuda";
  at::checkAllSameGPU(c, {input_t, filter_t, p_t});
  at::checkAllSameType(c, {input_t, filter_t, p_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto output = torch::empty({input.size(0),
                              filter.size(0),
                              dim.output_rows,
                              dim.output_cols},
                             input.options());
  auto lower = torch::empty_like(output);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(output.numel()), static_cast<int64_t>(512)));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "LMorph_forward", ([&] {
                               LMorphForwardCUDA<scalar_t> KERNEL_ARG4(grid, block, 0, stream)(
                                   output.numel(),
                                   dim,
                                   input.data_ptr<scalar_t>(),
                                   filter.data_ptr<scalar_t>(),
                                   p.data_ptr<scalar_t>(),
                                   lower.data_ptr<scalar_t>(),
                                   output.data_ptr<scalar_t>());
                             }));
  AT_CUDA_CHECK(cudaGetLastError());

  return {output, lower};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LMorph_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    const at::Tensor &lower,
    const at::Tensor &input,
    const at::Tensor &filter,
    const at::Tensor &p)
{
  AT_ASSERTM(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
  AT_ASSERTM(output.device().is_cuda(), "output must be a CUDA tensor");
  AT_ASSERTM(lower.device().is_cuda(), "lower must be a CUDA tensor");
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(filter.device().is_cuda(), "filter must be a CUDA tensor");
  AT_ASSERTM(p.device().is_cuda(), "p must be a CUDA tensor");

  auto dim = ValidDim(input.sizes(), filter.sizes(), p.sizes());

  at::TensorArg grad_output_t{grad_output, "grad_output", 1}, lower_t{lower, "lower", 2}, input_t{input, "input", 3}, filter_t{filter, "filter", 4}, p_t{p, "p", 5};

  at::CheckedFrom c = "LMorph_backward_cuda";
  at::checkAllSameGPU(c, {grad_output_t, lower_t, input_t, filter_t, p_t});
  at::checkAllSameType(c, {grad_output_t, lower_t, input_t, filter_t, p_t});

  at::cuda::CUDAGuard device_guard(grad_output.device());

  auto grad_input = at::zeros_like(input);
  auto grad_filter = at::zeros_like(filter);
  auto grad_p = at::zeros_like(p);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(grad_output.numel()), static_cast<int64_t>(512)));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "LMorph_backward", ([&] {
                               LMorphBackwardCUDA<scalar_t> KERNEL_ARG4(grid, block, 0, stream)(
                                   grad_output.numel(),
                                   dim,
                                   grad_output.data_ptr<scalar_t>(),
                                   output.data_ptr<scalar_t>(),
                                   lower.data_ptr<scalar_t>(),
                                   input.data_ptr<scalar_t>(),
                                   filter.data_ptr<scalar_t>(),
                                   p.data_ptr<scalar_t>(),
                                   grad_input.data_ptr<scalar_t>(),
                                   grad_filter.data_ptr<scalar_t>(),
                                   grad_p.data_ptr<scalar_t>());
                             }));
  AT_CUDA_CHECK(cudaGetLastError());

  return {grad_input, grad_filter, grad_p};
}
