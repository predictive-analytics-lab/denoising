import torch.nn as nn

from models.layers import GatedConvLayer, ConvLayer

PATCH_SIZE = 64


class SimpleDiscriminator(nn.Module):

    def __init__(self, args):
        super().__init__()

        layers = [ConvLayer(args.cnn_in_channels, 64, stride=1, normalize=False,
                            layer_activation=nn.LeakyReLU(0.1), num_classes=args.num_classes)]

        in_channels = layers[0].out_channels
        in_dim = PATCH_SIZE / layers[0].stride  # patch size is 64

        layer_params = [(64, 3, 2), (128, 3, 1), (128, 3, 2), (256, 3, 1), (256, 3, 2)]

        for out_channels, kernel_size, stride in layer_params:
            conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             normalize=False, layer_activation=nn.LeakyReLU(0.1),
                             num_classes=args.num_classes)
            layers.append(conv)
            in_channels = out_channels
            in_dim //= stride

        self.model = nn.ModuleList(layers)

        self.global_pooling = nn.AvgPool2d(int(in_dim))
        self.fc = nn.Linear(int(in_channels), 1)

    def forward(self, x):
        out = x

        for layer in self.model:
            out = layer(out)
        out = self.global_pooling(out)
        out = self.fc(out.view(x.size(0), -1))

        return out
