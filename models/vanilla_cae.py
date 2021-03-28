import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaCAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 out_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaCAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, #3 for mnist, 4 for mvtec 
                              stride= 2, #2 for mnist, 3 for mvtec 
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
                                       stride = 2, #2 for mnist, 3 for mvtec
                                       padding=1, #1 for mnist, 0 for mvtec
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
                                               kernel_size=3, #2 for mnist, 15 for mvtec
                                               stride=2,
                                               padding=1, #2 for mnist, 0 for mvtec
                                               output_padding=1), #0 for mnist
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], 
                                    out_channels= out_channels, #1 for mnist
                                    kernel_size= 3, 
                                    padding= 1),
                            # nn.Tanh())
                            nn.Sigmoid())
        
    #     self.weight_init()
    
    # def weight_init(self):

    #     def normal_init(m):
    #         if isinstance(m, (nn.Linear, nn.Conv2d)):
    #             m.weight.data.normal_(0,1)
    #             if m.bias.data is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             if m.bias.data is not None:
    #                 m.bias.data.zero_()
        
    #     for block in self._modules:
    #         try:
    #             for m in self._modules[block]:
    #                 normal_init(m)
    #         except:
    #             normal_init(block)

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
                      *args,
                      **kwargs) -> dict:
        """
        Computes the CAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]

        recons_loss =F.mse_loss(recons, input)

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

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]