import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3,
                 n_conv_layers=1,
                 n_input_channels=2,
                 n_output_channels=1,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(in_channels=n_input_channels, 
                          out_channels= n_output_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding= padding, bias=conv_bias),
                nn.Identity() if activation is None else activation()
                #nn.MaxPool1d(kernel_size=pooling_kernel_size) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.activation = activation()
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=1, length = 1000):
        return self.forward(torch.zeros((1, n_channels, length))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.activation(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
