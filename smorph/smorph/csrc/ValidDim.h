#pragma once

#include <torch/extension.h>

class ValidDim
{
public:
  long output_rows;
  long output_cols;
  long input_cols;
  long input_rows;
  long input_channels;
  long output_channels;
  long filter_cols;
  long filter_rows;
  long input_row_stride;
  long input_channel_stride;
  long input_batch_stride;
  long filter_row_stride;
  long filter_input_stride;
  long filter_output_stride;

  ValidDim(c10::IntArrayRef input,
           c10::IntArrayRef filter,
           c10::IntArrayRef p);
};
