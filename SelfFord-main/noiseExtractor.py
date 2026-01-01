
import math
import torch
import torch.nn as nn

def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))

def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()

def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)

def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    r"""
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features)==len(kernels))

    layers = list()
    for i in range(0,depth):
        if i==0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


class NoiseExtractor(nn.Module):
    r"""
        Network for extract the NoisePrint of an image,
        and then vectorize.
    """
    def __init__(self, nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None, dncnn_path=None):
        r"""
        :param nplanes_in: number of of input feature channels
        :param kernels: list of kernel size for convolution layers
        :param features: list of hidden layer feature channels
        :param bns: list of whether to add batchnorm layers
        :param acts: list of activations
        :param dilats: list of dilation factors
        :param bn_momentum: momentum of batchnorm
        :param padding: integer for padding (None for same padding)
        """
        super().__init__()

        depth = len(features)
        assert (len(features) == len(kernels))

        # construct DnCNN
        layers = list()
        for i in range(0, depth):
            if i == 0:
                in_feats = nplanes_in
            else:
                in_feats = features[i - 1]

            elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding,
                                     bias=not (bns[i]))
            conv_init(elem, act=acts[i])
            layers.append(elem)

            if bns[i]:
                elem = nn.BatchNorm2d(features[i], momentum=bn_momentum)
                batchnorm_init(elem, kernelsize=kernels[i])
                layers.append(elem)

            elem = make_activation(acts[i])
            if elem is not None:
                layers.append(elem)

        self.dnCNN = nn.Sequential(*layers)

        # construct the vectorization head
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.Dropout(0.1))

        # initialize head and DnCNN params
        self.initDnCNN(dncnn_path)
        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv3)
        conv_init(self.conv4)
        batchnorm_init(self.bn1, kernelsize=8)
        batchnorm_init(self.bn2, kernelsize=3)

    def initDnCNN(self, path):
        if path is not None:
            dat = torch.load(path, map_location=torch.device('cpu'))
            self.dnCNN.load_state_dict(dat)
        else:
            print('DnCNN is not inited')

    def forward(self, inputs):
        results = []
        for i in range(len(inputs)):
            y = self.dnCNN(inputs[i])
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.conv2(y)
            y = self.bn2(y)
            y = self.relu(y)

            y = self.conv3(y)
            y = self.bn3(y)
            y = self.relu(y)

            y = self.conv4(y)
            y = self.bn4(y)

            y = self.avgpool(y)
            y = torch.flatten(y, 1)
            y = self.fc(y)
            results.append(y)

        return results
