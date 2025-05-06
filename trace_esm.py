import os
from esm.pretrained import ESMC_300M_202412
import torch
import torch_neuronx
batch_size=1
def create_tokens(batch_size, seq_length=2048):
    # Create a single sequence: [0] + 2046 copies of 32 + [2]
    base_tokens = [0] + [32] * (seq_length - 2) + [2]
    # Repeat for the given batch size
    tokens = torch.tensor([base_tokens] * batch_size)
    return tokens
model = ESMC_300M_202412(device="cpu", use_flash_attn=False)
tokens = create_tokens(batch_size)
traced_model = torch_neuronx.trace(
    model, 
    tokens, 
    compiler_workdir='/home/ubuntu/environment/esm3_neuron/workdir/'
)
traced_model.save(f"./esmc-300m-inferentia_batch_size{batch_size}.pt")
