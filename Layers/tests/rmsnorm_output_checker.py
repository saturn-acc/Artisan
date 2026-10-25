import torch
import torch.nn as nn
from Layers.rmsnorm import rmsnorm
import torch.nn.functional as F


class test_benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_states = torch.randn(1, 256, device="mps")
        self.weights = torch.ones(256, device="mps")
        self.epsilon = 1e-5

    def test(self): 
        metal_output = rmsnorm(self.hidden_states, self.weights, self.epsilon)

        rms_norm_torch = torch.nn.functional.rms_norm(
        self.hidden_states,
        normalized_shape=(self.hidden_states.shape[-1],),
        weight=self.weights,
        eps=self.epsilon
        )


        correct = torch.allclose(
            metal_output,
            rms_norm_torch,
            atol=1e-5,
            rtol=1e-5
        )

        if correct == True:
            print("metal rmsnorm kernal output matches pytorch output ✅")
        else:
            print("rmsnorm metal kernal output does not match pytorch output ❌")

test_benchmark = test_benchmark()  
test_benchmark.test()    