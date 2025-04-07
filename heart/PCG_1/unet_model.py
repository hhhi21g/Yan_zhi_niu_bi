from .unet_parts import *

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

