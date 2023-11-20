# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import MLPBlock
from monai.utils import deprecated_arg

__all__ = ["VarAutoEncoder"]


class VarAutoEncoder(AutoEncoder):
    """
    Variational Autoencoder based on the paper - https://arxiv.org/abs/1312.6114

    Args:
        spatial_dims: number of spatial dimensions.
        in_shape: shape of input data starting with channel dimension.
        out_channels: number of output channels.
        latent_size: size of the latent variable.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        inter_channels: sequence of channels defining the blocks in the intermediate layer between encode and decode.
        inter_dilations: defines the dilation value for each block of the intermediate layer. Defaults to 1.
        num_inter_units: number of residual units for each block of the intermediate layer. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Examples::

        from monai.networks.nets import VarAutoEncoder

        # 3 layer network accepting images with dimensions (1, 32, 32) and using a latent vector with 2 values
        model = VarAutoEncoder(
            dimensions=2,
            in_shape=(32, 32),  # image spatial shape
            out_channels=1,
            latent_size=2,
            channels=(16, 32, 64),
            strides=(1, 2, 2),
        )

    see also:
        - Variational autoencoder network with MedNIST Dataset
          https://github.com/Project-MONAI/tutorials/blob/master/modules/varautoencoder_mednist.ipynb
    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:

        self.in_channels, *self.in_shape = in_shape

        self.latent_size = latent_size
        self.final_size = np.asarray(self.in_shape, dtype=int)
        if dimensions is not None:
            spatial_dims = dimensions

        super().__init__(
            spatial_dims,
            self.in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            inter_channels,
            inter_dilations,
            num_inter_units,
            act,
            norm,
            dropout,
            bias,
        )

        padding = same_padding(self.kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)  # type: ignore

        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.mu = nn.Linear(linear_size, self.latent_size)
        self.logvar = nn.Linear(linear_size, self.latent_size)
        self.decodeL = nn.Linear(self.latent_size, linear_size)

    def encode_forward(self, x: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:
        x = self.encode(x)
        x = self.intermediate(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decode_forward(self, z: torch.torch.Tensor, use_sigmoid: bool = True) -> torch.torch.Tensor:
        x = F.relu(self.decodeL(z))
        x = x.view(x.shape[0], self.channels[-1], *self.final_size)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def reparameterize(self, mu: torch.torch.Tensor, logvar: torch.torch.Tensor) -> torch.torch.Tensor:
        std = torch.exp(0.5 * logvar)

        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    def forward(self, x: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, torch.torch.Tensor, torch.torch.Tensor, torch.torch.Tensor]:
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_forward(z), mu, logvar, z


import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass


class MLPVAE(nn.Module):
    def __init__(self,
                 in_shape: int=(1,28,28),
                 latent_dim: int=32,
                 hidden_dims: List = [256, 256, 512, 32],
                 **kwargs) -> None:
        super(MLPVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []

        self.in_shape = in_shape
        hidden_dims = [math.prod(self.in_shape)] + hidden_dims

        # Build Encoder
        for h in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[h], hidden_dims[h+1]),
                    nn.BatchNorm1d(hidden_dims[h+1]),
                    nn.Tanh(),
                    )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        self.latent_shape = None
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for h in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[h], hidden_dims[h+1]),
                    nn.BatchNorm1d(hidden_dims[h+1]),
                    nn.Tanh(),
                    )
            )

        self.decoder = nn.Sequential(*modules)

    def encode_forward(self, x: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:
        x = self.encoder(x.flatten(start_dim=1))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode_forward(self, z: torch.torch.Tensor) -> torch.torch.Tensor:
        x = self.decoder(z)
        x = x.reshape(-1, *self.in_shape)
        return torch.sigmoid(x)

    def reparameterize(self, mu: torch.torch.Tensor, logvar: torch.torch.Tensor) -> torch.torch.Tensor:
        std = torch.exp(0.5 * logvar)
        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)
        return std.add_(mu)

    def forward(self, x: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, torch.torch.Tensor, torch.torch.Tensor, torch.torch.Tensor]:
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_forward(z), mu, logvar, z
    