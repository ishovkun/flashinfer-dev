#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace flashinfer::mamba::conversion {

inline __device__ float toFloat(float f) { return f; }

inline __device__ float toFloat(__half h) { return __half2float(h); }

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ float toFloat(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// No accuracy loss: int16_t range [-32768, 32767] fits exactly in float32
// (24-bit mantissa represents all integers up to 2^24 = 16M exactly).
inline __device__ float toFloat(int16_t val) { return static_cast<float>(val); }

inline __device__ void convertAndStore(float* output, float input) { *output = input; }

inline __device__ void convertAndStore(__half* output, float input) {
  *output = __float2half(input);
}

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ void convertAndStore(__nv_bfloat16* output, float input) {
  *output = __float2bfloat16(input);
}
#endif

inline __device__ void convertAndStore(int16_t* output, float input) {
  *output = static_cast<int16_t>(__float2int_rn(input));
}

}  // namespace flashinfer::mamba::conversion
