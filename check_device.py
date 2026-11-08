import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

print("\n--- Apple GPU Verification ---")

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# Test computation
x = torch.randn(1000, 1000, device=device)
y = x @ x

print("\nComputation successful!")
print("Output shape:", y.shape)
print("Device used:", y.device)