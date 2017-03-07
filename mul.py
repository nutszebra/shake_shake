import six
from chainer import cuda
from chainer import function


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Mul(function.Function):

    def __init__(self, train=False):
        self.train = train

    def forward(self, inputs):
        x1, x2 = inputs[:2]
        xp = cuda.get_array_module(x1)
        alpha = xp.ones(x1.shape, dtype=x1.dtype) * 0.5
        if self.train is True:
            for i in six.moves.range(len(alpha)):
                alpha[i] = xp.random.rand()
        return x1 * alpha + x2 * (xp.ones(x1.shape, dtype=x1.dtype) - alpha),

    def backward(self, inputs, grad_outputs):
        gx = grad_outputs[0]
        xp = cuda.get_array_module(gx)
        beta = xp.empty(gx.shape, dtype=gx.dtype)
        for i in six.moves.range(len(beta)):
            beta[i] = xp.random.rand()
        return gx * beta, gx * (xp.ones(gx.shape, dtype=gx.dtype) - beta)


def mul(x1, x2, train=False):
    return Mul(train)(x1, x2)
