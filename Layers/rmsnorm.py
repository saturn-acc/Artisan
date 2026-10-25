import torch
from pathlib import Path

METAL_PATH = Path(__file__).parent / "Kernals" / "rmsnorm.metal"

with open(METAL_PATH, "r") as f:
    source = f.read()

compiled_kernel = torch.mps.compile_shader(source)

def rmsnorm(hidden_state, rmsnorm_weight, epsilon):
    hidden_state = hidden_state.contiguous()
    rmsnorm_weight = rmsnorm_weight.contiguous()

    output = torch.empty_like(hidden_state)

    hidden_size = hidden_state.shape[-1]
    num_rows = hidden_state.numel() // hidden_size
    debug_value = torch.empty(1, device="mps", dtype=torch.float32)

    compiled_kernel.rmsnorm(
        hidden_state,
        output,
        rmsnorm_weight,
        hidden_size,
        epsilon,
        debug_value,
        threads=[num_rows * 256, 1, 1],
        group_size=[256, 1, 1]
       
    )
    print(debug_value.item())
    

    return output
    