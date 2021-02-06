#include "ValidDim.h"

ValidDim::ValidDim(c10::IntArrayRef input,
                   c10::IntArrayRef filter,
                   c10::IntArrayRef p)
{
    AT_ASSERTM(input.size() == 4, "input must be 4-dimensional, received ", input);
    AT_ASSERTM(filter.size() == 4, "filter must be 4-dimensional, received ", filter);
    AT_ASSERTM(p.size() == 2, "p must be 2-dimensional, received ", p);

    AT_ASSERTM(
        filter[1] == input[1],
        "filter depth must be the same as input depth (", filter[1], " != ", input[1], ")");
    AT_ASSERTM(p[1] == input[1],
               "p depth must be the same as input depth (", p[1], " != ", input[3], ")");
    AT_ASSERTM(p[0] == filter[0],
               "p out depth must be the same as filter out depth (", p[0], " != ", filter[0], ")");

    auto padding_row = (filter[2] >> 1) << 1;
    auto padding_col = (filter[3] >> 1) << 1;

    output_rows = input[2] - padding_row;
    output_cols = input[3] - padding_col;    input_channels = input[1];

    input_rows = input[2];    input_cols = input[3];

    output_channels = filter[0];    filter_rows = filter[2];

    filter_cols = filter[3];
    input_row_stride = input_cols;
    input_channel_stride = input_rows * input_row_stride;
    input_batch_stride = input_channels * input_channel_stride;
    filter_row_stride = filter_cols;
    filter_input_stride = filter_rows * filter_row_stride;
    filter_output_stride = input_channels * filter_input_stride;
}
