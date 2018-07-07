#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace Lotus {
namespace Cuda {

template <typename T>
void TileImpl(
    const size_t shape_rank,
    const fast_divmod* input_shape,
    const int64_t* input_strides,
    const T* input_data,
    const fast_divmod* fdm_output_strides,
    T* output_data,
    const size_t N);

}  // namespace Cuda
}  // namespace Lotus