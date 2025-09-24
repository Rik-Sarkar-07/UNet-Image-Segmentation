from my_models.unet import UNet

def get_model(model_name, num_classes, depth, num_channels):
    if model_name == 'unet':
        return UNet(num_classes=num_classes, depth=depth, num_channels=num_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")