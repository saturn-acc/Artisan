#include <metal_stdlib>
using namespace metal;
#pass in the epsilon in the python wrapper 
kernel void rmsnorm(const device float* embeds, device float*norm_embeds, constant uint& hidden_size, constant float& epsilon) {

  uint tid [[thread_position_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]],
  uint num_threads [[threads_per_threadgroup]],
  uint simd_lane_id [[thread_index_in_simdgroup]],
  uint simd_group_id [[simdgroup_index_in_threadgroup]]

  float partial_sum = 0.0;
  for (uint i = tid; i < hidden_size; i += num_threads) {
      float x = embeds[row * hidden_size + i];
      partial_sum += x * x;
  }

  float simd_total = simd_sum(partial_sum); 
  threadgroup float simd_sums[8]; 

  if (simd_lane_id == 0) { 
    simd_sums[simd_group_id] = simd_total;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0 && simd_group_id == 0) {
    float total_sum = 0.0;
    for (uint i = 0; i < 8; i++)  {
      total_sum += simd_sums[i];
    }
  }

  float divided = total_sum / hidden_size


}
