import torch 
import torch.nn as nn
from Layers.rmsnorm import rmsnorm
import time
import torch.nn.functional as F

class benchmark_rms(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_states = torch.randn(128, 2048, device="mps")
        self.weight = torch.ones(2048, device="mps")
        self.epsilon = 1e-5

    def benchmark(self):
        for _ in range(50):
            rmsnorm(self.hidden_states, self.weight, self.epsilon)
        torch.mps.synchronize()

        start = time.perf_counter()

        runs = 1000

        for _ in range(runs):
            rmsnorm(self.hidden_states, self.weight, self.epsilon)
        torch.mps.synchronize()

        end = time.perf_counter()

        
        average_metelkernal_ms = ((end - start) / runs) * 1000

        for _ in range(50):
             rms_norm_torch = torch.nn.functional.rms_norm(
            self.hidden_states,
            normalized_shape=(self.hidden_states.shape[-1],),
            weight=self.weight,
            eps=self.epsilon
            )
        torch.mps.synchronize()


        start = time.perf_counter()     

        for _ in range(runs):
            rms_norm_torch = torch.nn.functional.rms_norm(
            self.hidden_states,
            normalized_shape=(self.hidden_states.shape[-1],),
            weight=self.weight,
            eps=self.epsilon
            )

        torch.mps.synchronize()

        end = time.perf_counter()

        average_torchrmsnorm_ms = ((end - start) / runs) * 1000

        print(f"average miliseconds for metal rmsnorm", average_metelkernal_ms)
        print(f"average miliseconds for pytorch rmsnorm", average_torchrmsnorm_ms)
            

benchmark = benchmark_rms()
benchmark.benchmark()

