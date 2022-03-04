from typing import List

import torch
from torch import nn
from torch.nn import functional as F


# Modified from https://github.com/AntixK/PyTorch-VAE
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright Anand Krishnamoorthy Subramanian 2020
# anandkrish894@gmail.com


class ConvResBlock(nn.Module):
    # Adapted from https://github.com/ermongroup/ncsn/blob/master/models/scorenet.py
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels)
        )
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.final_act = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, inputs):
        h = self.main(inputs)
        res = self.residual(inputs)
        h += res
        return self.final_act(h)


class DeconvResBlock(nn.Module):
    # Adapted from https://github.com/ermongroup/ncsn/blob/master/models/scorenet.py
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels)
        )
        self.residual = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_channels),
        )

        self.final_act = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, inputs):
        h = self.main(inputs)
        res = self.residual(inputs)
        h += res
        return self.final_act(h)


class BasicVAE3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 ) -> None:
        super(BasicVAE3d, self).__init__()

        self.latent_dim = latent_dim
        self.n_channels = in_channels

        modules = []
        hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(ConvResBlock(in_channels, h_dim))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 2**3, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 2**3, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 2**3)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(DeconvResBlock(hidden_dims[i], hidden_dims[i + 1]))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=self.n_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, initial: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param initial: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(initial)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, initial: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(initial)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), initial, mu, log_var]

    def loss_function(self, recons, initial, mu, log_var, kld_weight) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons_loss = F.mse_loss(recons, initial)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
