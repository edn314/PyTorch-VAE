import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from math import exp


class MS_SSIM_CAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 out_channels: int,
                 hidden_dims: List = None,
                 window_size: int = 11,
                 size_average: bool = True,
                 **kwargs) -> None:
        super(MS_SSIM_CAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, 
                              stride= 2,
                              padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, out_channels=h_dim,
                              kernel_size= 3,
                              stride= 1,
                              padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    # nn.Conv2d(h_dim, out_channels=h_dim,
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(h_dim, out_channels=h_dim,
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(h_dim, out_channels=h_dim,
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    # nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) 

        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3, 
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
                              kernel_size= 3,
                              stride= 1,
                              padding  = 1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    # nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
                    #           kernel_size= 3,
                    #           stride= 1,
                    #           padding  = 1),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    # nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], 
                                    out_channels= out_channels,
                                    kernel_size= 3, 
                                    padding= 1),
                            # nn.Tanh())
                            nn.Sigmoid())

        self.mssim_loss = MSSIM(self.in_channels,
                                window_size,
                                size_average)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # ##### DEBUG #####
        # print("----Start enc layers----")
        # x = input
        # print(x.shape)
        # for mod in self.encoder:
        #     x = mod(x)
        #     print(x.shape)
        # import pdb; pdb.set_trace()

        result = self.encoder(input)

        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        
        # ### DEBUG #####
        # print("----Start dec layers----")
        # x = result
        # print(x.shape)
        # for mod in self.decoder:
        #     x = mod(x)
        #     print(x.shape)
        # import pdb; pdb.set_trace()

        result = self.decoder(z)
        result = self.final_layer(result)

        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        x = self.encode(input)
        return  [self.decode(x), input] # for convolutional autoencoder

    def loss_function(self,
                      *args: Any,
                      **kwargs) -> dict:
        """
        Computes the CAE MS-SSIM loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]

        recons_loss = self.mssim_loss(recons, input)

        # l2_loss = F.mse_loss(recons, input)
        # l1_loss = F.l1_loss(recons, input)
        # huber_loss = F.smooth_l1_loss(recons, input)

        # alpha = 0.7
        # loss = alpha*recons_loss + (1-alpha)*huber_loss 

        loss = recons_loss

        return {'loss': loss}

    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.cuda(current_device)

    #     samples = self.decode(z)
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class MSSIM(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 window_size: int=11,
                 size_average:bool = True) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)

        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size:int, sigma: float) -> Tensor:
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1: Tensor,
             img2: Tensor,
             window_size: int,
             in_channel: int,
             size_average: bool) -> Tensor:

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = img1.max() - img1.min()
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


