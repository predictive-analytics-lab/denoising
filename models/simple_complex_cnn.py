import torch
import torch.nn as nn

from models import GatedConvLayer
from models.complex_layers import ComplexGatedConvLayer


class ComplexGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        num_hidden_channels = args.cnn_hidden_channels // 2
        # Input layer
        layers = [ComplexGatedConvLayer(args.cnn_in_channels, num_hidden_channels,
                                        local_condition=args.iso, num_classes=args.num_classes)]
        # Hidden layers
        for _ in range(args.cnn_hidden_layers):
            layers.append(ComplexGatedConvLayer(num_hidden_channels, num_hidden_channels,
                                                local_condition=args.iso, num_classes=args.num_classes))

        layers.append(ComplexGatedConvLayer(num_hidden_channels, args.cnn_in_channels,
                                            layer_activation=None, normalize=False,
                                            local_condition=args.iso, num_classes=args.num_classes))

        self.model = nn.ModuleList(layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = torch.rfft(x, onesided=False, signal_ndim=2)

        for layer in self.model:
            out = layer(out, c, class_labels)

        # inverse fourier transform
        out = torch.irfft(out, signal_ndim=2, onesided=False, signal_sizes=x.shape[2:])

        if self.residual:   # learn noise residual
            out = out + x

        return out
