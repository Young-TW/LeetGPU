#!/usr/bin/env python3
"""Local check for #41 Simple Inference using pytorch-rocm."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
from simple_inference import solve  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Linear(2, 2).to(dev)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.5, 1.0], [1.5, 0.5]]))
    model.bias.copy_(torch.tensor([0.1, 0.2]))

inp = torch.tensor([[1.0, 2.0]], device=dev)
out = torch.empty(1, 2, device=dev)
solve(inp, model, out)

want = torch.tensor([[2.6, 2.7]], device=dev)
if torch.allclose(out, want, atol=1e-4):
    print("PASS")
    sys.exit(0)
print(f"FAIL: got {out.tolist()} want {want.tolist()}")
sys.exit(1)
