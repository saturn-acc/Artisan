#include <metal_stdlib>
using namespace metal;

kernal void rmsnorm(const device float* embeds, device float*norm_embeds, constant uint& hidden_size, uint tid [[thread_position_in_grid]]) {
  for (unint i = tid; i < hidden_size; i += num_threads) {
      float x = input[row * hidden_size + i];
      float squared= x * x;
    }   
}
