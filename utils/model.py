import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=4, stride=2,padding=1,
         batch_norm =True):
    '''

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: kernel size input
    :param stride: stride number
    :param padding: padding
    :param batch_norm: batch norm boolean value
    :return:Sequential layer
    '''
    layers = []
    conv_layers = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding)
    layers.append(conv_layers)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
             batch_norm=True):
    '''
    Creates a transpose convolution layer, with optional batch normalization option
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: kernel size input
    :param stride: stride number
    :param padding: padding
    :param batch_norm: batch norm boolean value
    :return:Sequential layer
    '''
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,stride, padding,
                                     bias=True))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    '''Class for defining residual bocks(skip connections)'''

    def __init__(self, conv_dim):
        '''
        param conv_dim: convolution dimension for input and output
        '''
        super(ResidualBlock, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3,
                          stride = 1, padding=1, batch_norm=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3,
                          stride=1, padding=1, batch_norm=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_1 = self.conv1(x)
        out_1 = self.relu(out_1)
        out_2 = x + self.conv2(out_1)
        return out_2

class Discriminator(nn.Module):
    '''Class for defining Discriminator architecture'''
    def __init__(self, conv_dim = 64):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(in_channels=3, out_channels = conv_dim, batch_norm=False)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4)
        self.conv4 = conv(in_channels=conv_dim*4, out_channels=conv_dim*8)

        self.conv5 = conv(in_channels=conv_dim*8, out_channels=1, batch_norm=False)


    def forward(self, x):
        x = x.view(-1,3,256,256)

        out = F.leaky_relu(self.conv1(x),0.2)
        out = F.leaky_relu(self.conv2(out),0.2)
        out = F.leaky_relu(self.conv3(out),0.2)
        out = F.leaky_relu(self.conv4(out),0.2)
        out = self.conv5(out)


        return out

class Generator(nn.Module):
    '''Class for defining Generator Architecture'''
    def __init__(self,conv_dim=64, n_res_blocks = 6):
        super(Generator, self).__init__()

        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        res_layers = []
        for i in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))
            
        self.res_block = nn.Sequential(*res_layers)
        
        self.deconv1 = deconv(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=4)
        self.deconv2 = deconv(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=4)
        self.deconv3 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4)

    def forward(self, x):
        x = x.view(-1,3,256,256)

        out = F.leaky_relu(self.conv1(x),0.2)
        out = F.leaky_relu(self.conv2(out),0.2)
        out = F.leaky_relu(self.conv3(out),0.2)

        out = self.res_block(out)

        out = F.leaky_relu(self.deconv1(out), 0.2)
        out = F.leaky_relu(self.deconv2(out), 0.2)
        out = torch.tanh(self.deconv3(out))

        return out
    