import torch
from my_models.unet import UNet
from torchprofile import profile_macs

def benchmark_model(model_name, num_classes, depth, num_channels, img_size):
    model = UNet(num_classes=num_classes, depth=depth, num_channels=num_channels)
    inputs = torch.randn(1, 3, img_size, img_size)
    macs = profile_macs(model, inputs)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, Parameters: {params/1e6:.2f}M, FLOPs: {macs/1e9:.2f}G")