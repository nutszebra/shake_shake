import six
from chainer import cuda
from chainer import function


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Mul(function.Function):

    def __init__(self):
        pass

    def forward(self, inputs):
        x1, x2 = inputs[:2]
        xp = cuda.get_array_module(x1)
        alpha = xp.empty(x1.shape, dtype=x1.dtype)
        for i in six.moves.range(len(alpha)):
            alpha[i] = xp.random.rand()
        return x1 * alpha + x2 * (xp.ones(x1.shape, dtype=x1.dtype) - alpha),

    def backward(self, inputs, grad_outputs):
        gx1, gx2 = inputs[:2]
        xp = cuda.get_array_module(gx1)
        beta = xp.empty(gx1.shape, dtype=gx1.dtype)
        for i in six.moves.range(len(beta)):
            beta[i] = xp.random.rand()
        return gx1 * beta, gx2 * (xp.ones(gx1.shape, dtype=gx1.dtype) - beta)


def mul(x1, x2):
    func = Mul()
    return func(x1, x2)
