import pytorch as torch 
import pathlib as Path
METAL_PATH = Path(__file__).parent.parent / "Kernels" / "rmsnorm.metal"

with open(METAL_PATH, "r") as f:
    source = f.read()

compiled_kernal = torch.compile.metal_shader(source)

def rmsnorm (hidden_state, rmsnorm_weight, epsilon):
    output = torch.empty_like(hidden_state)
    hidden_size = hidden_state.shape[-1]

    compiled_kernal.rmsnorm(
        hidden_state,
        output,
        hidden_size,
        epsilon,
        threads=256
        
    )
    return output


