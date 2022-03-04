# From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

# Basic GAN with 3D voxel data


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class BasicGenerator(nn.Module):
    def __init__(self, nc, nz, ngf, leak=0.1):
        """
        :param nc: Number of channels in the training images. For color images this is 3
        :param nz: Size of z latent vector (i.e. size of generator input)
        :param ngf: Size of feature maps in generator
        :param leak: LeakyReLU slope
        """
        super(BasicGenerator, self).__init__()
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv3d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose3d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.Conv3d(in_channels=ngf, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
            # state size. (nc) x 16 x 16
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x) * 2 - 1


class BasicDiscriminator(nn.Module):
    def __init__(self, nc, ndf, leak=0.1):
        """
        :param nc: Number of channels in the training images. For color images this is 3
        :param ndf: Size of feature maps in discriminator
        :param leak: LeakyReLU slope
        """
        super(BasicDiscriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 16 x 16
            sn(nn.Conv3d(in_channels=nc, out_channels=ndf, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ndf) x 16 x 16
            sn(nn.Conv3d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ndf*2) x 8 x 8
            sn(nn.Conv3d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=1, padding=0, bias=True)),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ndf*4) x 8 x 8
            sn(nn.Conv3d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(leak, inplace=True),
            # state size. (ndf*8) x 4 x 4
            sn(nn.Conv3d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x).view(-1)
