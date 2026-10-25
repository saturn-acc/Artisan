#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(

const device float* embeds  [[buffer(0)]], 
device float* norm_embeds  [[buffer(1)]], 
device float* norm_weight  [[buffer(2)]], 
constant uint& hidden_size  [[buffer(3)]], 
constant float& epsilon  [[buffer(4)]],
device float* debug_values [[buffer(5)]],
uint tid [[thread_position_in_threadgroup]],
uint row [[threadgroup_position_in_grid]],
uint num_threads [[threads_per_threadgroup]],
uint simd_lane_id [[thread_index_in_simdgroup]],
uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  float partial_sum = 0.0;
  for (uint i = tid; i < hidden_size; i += num_threads) {
      float x = embeds[row * hidden_size + i];
      partial_sum += x * x;
  }

  float simd_total = simd_sum(partial_sum); 
  threadgroup float simd_sums[8]; 
  threadgroup float final_sum;

  if (simd_lane_id == 0) { 
    simd_sums[simd_group_id] = simd_total;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0 && simd_group_id == 0) {
    float total_sum = 0.0;
    for (uint i = 0; i < 8; i++)  {
      total_sum += simd_sums[i];
    }
    final_sum = total_sum;

  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float x = final_sum / hidden_size;
  x = x + epsilon;
  float normilized_value = sqrt(x);

  if (tid == 0) {
    debug_values[row] = normilized_value;
  }

   for (uint i = tid; i < hidden_size; i += num_threads) {
    float x = embeds[row * hidden_size + i];
    float y = norm_weight[i];

    float z = (x / normilized_value) * y;
    
    norm_embeds[row * hidden_size + i] = z;
    
  }
  
}
