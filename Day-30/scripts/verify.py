import torch

print("CUDA available:", torch.cuda.is_available())  # Should print: True
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
