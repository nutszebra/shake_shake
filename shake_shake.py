import six
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict
from mul import mul


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class ReLU_Conv_BN_ReLU_Conv_BN(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(ReLU_Conv_BN_ReLU_Conv_BN, self).__init__(
            conv1=L.Convolution2D(in_channel, out_channel, filter_size[0], stride[0], pad[0]),
            bn1=L.BatchNormalization(out_channel),
            conv2=L.Convolution2D(out_channel, out_channel, filter_size[1], stride[1], pad[1]),
            bn2=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        self.conv2.W.data = self.weight_relu_initialization(self.conv2)
        self.conv2.b.data = self.bias_initialization(self.conv2, constant=0)

    def count_parameters(self):
        count = functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        count += functools.reduce(lambda a, b: a * b, self.conv2.W.data.shape)
        return count

    def __call__(self, x, train=False):
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x, test=not train)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, test=not train)
        return x


class Double(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel):
        out_channel1 = int(out_channel) / 2
        out_channel2 = out_channel - out_channel1
        super(Double, self).__init__(
            conv1=L.Convolution2D(in_channel, int(out_channel1), 1, 1, 0),
            conv2=L.Convolution2D(in_channel, int(out_channel2), 1, 1, 0),
            bn=L.BatchNormalization(out_channel),
        )

    def __call__(self, *args, **kwargs):
        x = args[0]
        train = kwargs['train']
        x1 = self.conv1(F.average_pooling_2d(x, 1, 2, 0))
        x2 = self.conv2(F.average_pooling_2d(self.zero_pads(self.zero_pads(x, 1, 2), 1, 3), 1, 2, 0)[:, :, 1:, 1:])
        return self.bn(F.concat((x1, x2), axis=1), test=not train)

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        self.conv2.W.data = self.weight_relu_initialization(self.conv2)
        self.conv2.b.data = self.bias_initialization(self.conv2, constant=0)

    def count_parameters(self):
        count = functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        count += functools.reduce(lambda a, b: a * b, self.conv2.W.data.shape)
        return count

    def zero_pads(self, x, pad, where, dtype=np.float32):
        sizes = list(x.data.shape)
        sizes[where] = pad
        pad_mat = self.prepare_input(np.zeros(sizes, dtype=dtype), volatile=x.volatile)
        if not type(x.data) == np.ndarray:
            pad_mat.to_gpu()
        return F.concat((pad_mat, x), axis=where)


class DoNothing(nutszebra_chainer.Model):

    def __init__(self):
        super(DoNothing, self).__init__()

    def __call__(self, *args, **kwargs):
        return args[0]

    def count_parameters(self):
        return 0


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, branch_num=2, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), identity=DoNothing()):
        super(ResBlock, self).__init__()
        modules = []
        for i in six.moves.range(branch_num):
            modules.append(('branch{}'.format(i), ReLU_Conv_BN_ReLU_Conv_BN(in_channel, out_channel, filter_size, stride, pad)))
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.branch_num = branch_num
        self.identity = identity

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules])) + self.identity.count_parameters()

    def __call__(self, x, train=False):
        branches = []
        for i in six.moves.range(self.branch_num):
            branches.append(self['branch{}'.format(i)](x, train=train))
        x = self.identity(x, train=train)
        return mul(*branches, train=train) + x


class ShakeShake(nutszebra_chainer.Model):

    def __init__(self, category_num, out_channels=(64, 128, 256), N=(4, 4, 4), branch_num=2):
        super(ShakeShake, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, out_channels[0], 3, 1, 1))]
        in_channel = out_channels[0]
        strides = [[(1, 1) for i in six.moves.range(N[ii])] for ii in six.moves.range(len(out_channels))]
        identities = [[DoNothing() for i in six.moves.range(N[ii])] for ii in six.moves.range(len(out_channels))]
        modules += [('double1', Double(64, 128))]
        modules += [('double2', Double(128, 256))]
        strides[1][0], identities[1][0] = (2, 1), modules[-2][1]
        strides[2][0], identities[2][0] = (2, 1), modules[-1][1]
        for i in six.moves.range(len(out_channels)):
            for n in six.moves.range(N[i]):
                modules.append(('res_block{}_{}'.format(i, n), ResBlock(in_channel, out_channels[i], branch_num, (3, 3), strides[i][n], (1, 1), identities[i][n])))
                in_channel = out_channels[i]
        modules.append(('linear', BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.out_channels = out_channels
        self.N = N
        self.branch_num = branch_num
        self.name = 'shake_shake|{}|{}|'.format(category_num, branch_num)
        self.name = '{}{}|'.format(self.name, '_'.join([str(i) for i in out_channels]))
        self.name = '{}{}|'.format(self.name, '_'.join([str(i) for i in N]))

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        for name, link in self.modules[1:]:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        for name, link in self.modules[1:]:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = F.relu(self.conv1(x))
        for i in six.moves.range(len(self.out_channels)):
            for n in six.moves.range(self.N[i]):
                h = self['res_block{}_{}'.format(i, n)](h, train=train)
        h = self.linear(h, train=train)
        batch, categories, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
