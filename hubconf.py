from my_models.unet import UNet

def unet(num_classes=19, depth=5, num_channels=64, **kwargs):
    return UNet(num_classes=num_classes, depth=depth, num_channels=num_channels, **kwargs)