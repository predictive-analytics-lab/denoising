"""DenseNet"""
import torch
import torch.nn as nn

from models import ConvLayer, GatedConvLayer


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block
    """
    def __init__(self, nc, gc=32, kernel_size=3, local_condition=True, learn_beta=False, beta=0.2):
        super().__init__()

        self.conv1 = ConvLayer(nc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv2 = ConvLayer(nc + gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv3 = ConvLayer(nc + 2 * gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv4 = ConvLayer(nc + 3 * gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))

        if learn_beta:
            self.conv5 = ConvLayer(nc + 4 * gc, nc, kernel_size=kernel_size, normalize=False,
                                   layer_activation=None)
        else:
            self.conv5 = GatedConvLayer(nc + 4 * gc, nc, kernel_size=kernel_size,
                                        local_condition=local_condition, conv_residual=False,
                                        normalize=False, layer_activation=None)

        self.beta = beta
        if learn_beta and local_condition:
            self.cond_beta = nn.Linear(1, nc, bias=False)
        else:
            self.register_parameter('cond_beta', None)

    def forward(self, x, c=None, class_labels=None):
        x1 = self.conv1(x, c)
        x2 = self.conv2(torch.cat((x, x1), 1), c)
        x3 = self.conv3(torch.cat((x, x1, x2), 1), c)
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1), c)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1), c)

        if self.cond_beta is not None:
            beta = self.cond_beta(c.view(-1, 1)).sigmoid()[..., None][..., None]
        else:
            beta = self.beta

        return x5.mul(beta) + x


class RDDB(nn.Module):
    """
    Residual in Residual Dense Block
    """
    def __init__(self, nc, gc=32, kernel_size=3, local_condition=True, learn_beta=False, beta=0.2):
        super().__init__()

        self.rdb1 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, learn_beta, beta)
        self.rdb2 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, learn_beta, beta)
        self.rdb3 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, learn_beta, beta)

        self.beta = beta
        if learn_beta and local_condition:
            self.cond_beta = nn.Linear(1, nc, bias=False)
        else:
            self.register_parameter('cond_beta', None)

    def forward(self, x, c=None, class_labels=None):
        out = self.rdb1(x, c, class_labels)
        out = self.rdb2(out, c, class_labels)
        out = self.rdb3(out, c, class_labels)

        if self.cond_beta is not None:
            beta = self.cond_beta(c.view(-1, 1)).sigmoid()[..., None][..., None]
        else:
            beta = self.beta

        return out.mul(beta) + x


class DenseGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()
        gc = 32
        # Input layer
        layers = [ConvLayer(args.cnn_in_channels, args.cnn_hidden_channels, normalize=False,
                            layer_activation=None)]

        # Hidden layers
        for _ in range(args.cnn_hidden_layers):
            layers.append(RDDB(args.cnn_hidden_channels, gc, local_condition=args.iso,
                               learn_beta=args.learn_beta, beta=0.2))
        # Output layer
        layers.append(ConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, normalize=False,
                                layer_activation=None))
        self.model = nn.ModuleList(layers)

        # init
        def _init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                module.weight.data *= 0.1
            elif isinstance(module, nn.Linear):
                module.weight.data *= 0.1

        self.apply(_init_weights)

        self.residual = args.residual

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)

        if self.residual:   # learn noise residual
            out = out + x

        return out
