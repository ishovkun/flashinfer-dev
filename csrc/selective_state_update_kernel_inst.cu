#include <flashinfer/mamba/selective_state_update.cuh>

#include "selective_state_update_config.inc"

namespace flashinfer::mamba {

template void invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(
    SelectiveStateUpdateParams&, cudaStream_t);

namespace mtp {
template void invokeSelectiveStateUpdateMTP<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(
    SelectiveStateMTPParams&, cudaStream_t);
}  // namespace mtp

}  // namespace flashinfer::mamba
