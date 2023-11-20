import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms, utils
from typing import List, Callable, Any, Optional, Sequence, Tuple, Union
from vae import VarAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--output_results', type=str, default='output_results_mlp')
parser.add_argument('--output_model', type=str, default='output_model_mlp')
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=128)
parser.add_argument('--kl_beta', type=float, default=1.0)
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id>-1 else 'cpu')

def scale_01(x):
    x = x - torch.min(x)
    x = x / (torch.max(x) + 1e-5)
    return x

if args.dataset == 'mnist':
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
                        	train=True, transform=transforms.ToTensor(), download=True),
                        	batch_size=args.batch_size_test, shuffle=False, num_workers=4, drop_last=True)
	test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
                        	train=False, transform=transforms.ToTensor(), download=True), 
                        	batch_size=args.batch_size_test, shuffle=False, num_workers=4, drop_last=True)
	args.c, args.w, args.h = 1, 28, 28


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

    def decode_forward(self, z: torch.torch.Tensor, use_sigmoid: bool = True) -> torch.torch.Tensor:
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
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

kl_betas = [1.0]
output_results = args.output_results  
output_model = args.output_model
for kl in kl_betas:
    # vae = VarAutoEncoder(dimensions=2,
    #         in_shape=(args.c, args.w, args.h),  # image spatial shape
    #         out_channels=1,
    #         latent_size=32,
    #         channels=(2, 4, 8),
    #         strides=(1, 2, 2),).to(device)
    vae = MLPVAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=200)
    losses = AverageMeter('generator loss')
    output_results = os.path.join(args.output_results, args.dataset,  f"kl_{kl}")
    output_model =  os.path.join(args.output_model,  args.dataset, f"kl_{kl}")
    os.makedirs(output_results, exist_ok=True)
    os.makedirs(output_model, exist_ok=True)
    for epoch in range(args.max_epochs):
        for i, (data, label) in enumerate(train_loader, 0):
            data = data.to(device)
            x, mu, logvar, z = vae(data)
            recons_loss = F.mse_loss(x, data)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            loss = recons_loss + kl * kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss)
        print(f"Training Epoch {epoch}", losses)
        if (epoch+1) % args.save_every== 0:
            grid = utils.make_grid(x, normalize=True)
            utils.save_image(grid, os.path.join(output_results, f"reconstruction_{epoch+1}.png"))
            grid = utils.make_grid(vae.decode_forward(torch.randn_like(z)), normalize=True)
            utils.save_image(grid, os.path.join(output_results, f"sampled_{epoch+1}.png"))
            model_path = os.path.join(output_model, f"{epoch+1}.pt")
            torch.save({'vae':vae.state_dict(), 'optimizer':optimizer.state_dict()}, model_path)
        scheduler.step()


