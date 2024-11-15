import torch
import numpy as np

from torch.nn import functional as F
from torch import optim, nn
import lightning as L
import pytorch_lightning as pl
import mltools.layers.torch as layers

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


_INPUT_CHANNELS: int = 3
_OUTPUT_CHANNELS: int = 3
_INITIALIZER = torch.nn.init.normal_

_DOWN_CONV2D_KWARGS: dict = dict(
    kernel_size = 4,
    stride = 2,
    bias = False,
    # padding = 'same',
    padding = 1,
)
_DOWN_ACTIVATION = torch.nn.LeakyReLU()

_UP_CONV2D_KWARGS: dict = dict(
    kernel_size = 4,
    stride = 2,
    bias = False,
    # padding = 'same',
    padding = 1,
)
_UP_ACTIVATION = torch.nn.LeakyReLU()

_LAST_CONV2D_KWARGS: dict = dict(
    kernel_size = 4,
    stride = 2,
    bias = True,
    padding = 'same',
)
_LAST_ACTIVATION = torch.nn.Sigmoid()



class _EncoderBlock(nn.Module):
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layer_kwargs: dict = _DOWN_CONV2D_KWARGS,
            batchnorm: bool = True,
            activation: callable = _DOWN_ACTIVATION,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = []
        self.layers.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **layer_kwargs,
            ),
        )
        if batchnorm:
            self.layers.append(
                nn.BatchNorm2d(num_features=self.out_channels),
            )
        self.layers.append(activation)
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
    


class _DecoderBlock(nn.Module):
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layer_kwargs: dict = _UP_CONV2D_KWARGS,
            dropout: bool = True,
            activation: callable = _UP_ACTIVATION,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **layer_kwargs,
            ),
        )
        if dropout:
            self.layers.append(
                nn.Dropout(),
            )
        self.layers.append(activation)
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
    
    
    
class Pix2Pix(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4, filters: list[int] = [64, 128, 256, 512]):
        super(Pix2Pix, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        channels = in_channels
        skip_chans = []
        for i in range(num_levels):
            # self.encoder.append(_EncoderBlock(channels, channels*2))
            # channels *= 2
            self.encoder.append(_EncoderBlock(channels, filters[i]))
            print(f"Encoder {i+1}: in_channels = {channels}, out_channels = {filters[i]}")
            skip_chans.append(filters[i])
            print(f"    Skip {i+1}: channels = {skip_chans[i]}")
            channels = filters[i]

        # Middle convolutional block
        self.middle_conv = _EncoderBlock(channels, channels*2)
        print(f"Latent: in_channels = {channels}, out_channels = {channels*2}")
        channels *= 2

        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(num_levels):
            # if i == 0:
            self.decoder.append(_DecoderBlock(channels, channels//2 + skip_chans[-(i - 1)]))
            print(f"Decoder {i+1}: in_channels = {channels}, out_channels = {channels//2 + skip_chans[-i - 1]}")
            
            # else:
            #     self.decoder.append(_DecoderBlock(channels, channels))
            channels //= 2

        # Final convolution
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        print('Downsampling starts')
        for encoder_block in self.encoder:
            x = encoder_block(x)
            print(x.shape)
            skip_connections.append(x)
        print('Downsampling ends')


        # Middle
        print('Latent space starts')
        x = self.middle_conv(x)
        print(x.shape)
        print('Latent space ends')
        

        # Decoder
        print('Upsampling starts')
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            skip = skip_connections[-(i+1)]
           
            print(f"x: {x.shape}, skip: {skip.shape}")
            
            # Resize x if necessary (in case of odd dimensions)
            # if x.shape != skip.shape:
            #     x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat((x, skip), dim=1)
            print(x.shape)
            # x = _EncoderBlock(x.shape[1], x.shape[1]//2)(x)  # Additional conv block after skip connection
        print('Upsampling ends')


        # Final convolution
        x = self.final_conv(x)

        return x