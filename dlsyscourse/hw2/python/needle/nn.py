"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype), (1, out_features)))
        self.need_bias = bias
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = ops.matmul(X, self.weight)
        if self.need_bias:
            Y += ops.broadcast_to(self.bias, Y.shape)
        return Y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, n = logits.shape[0], logits.shape[1]
        one_hot_y = init.one_hot(n, y)
        return (ops.summation(ops.logsumexp(logits, axes=(1,))) - ops.summation(logits * one_hot_y)) / batch_size
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)

        if self.training:
            x_mean = ops.summation(x, axes=0) / batch_size
            batch_mean = ops.reshape(x_mean, (1, -1)).broadcast_to(x.shape)
            diff = x - batch_mean
            x_vars = ops.summation(diff ** 2, axes=0) / batch_size
            batch_vars = ops.reshape(x_vars, (1, -1)).broadcast_to(x.shape)
            x_norm = diff / ((batch_vars + self.eps) ** 0.5)        
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_vars.detach()
            # batch_mean = ops.summation(x, axes=0) / batch_size
            # self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            # batch_mean = ops.reshape(batch_mean, (1, -1)).broadcast_to(x.shape)
            # batch_vars = ops.summation((x - batch_mean) ** 2, axes=0) / batch_size
            # self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_vars
            # batch_vars = ops.reshape(batch_vars, (1, -1)).broadcast_to(x.shape)
        else:
            batch_mean = ops.reshape(self.running_mean, (1, -1)).broadcast_to(x.shape)
            batch_vars = ops.reshape(self.running_var, (1, -1)).broadcast_to(x.shape)
            x_norm = (x - batch_mean) / ((batch_vars + self.eps) ** 0.5)        
        return weight * x_norm + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)).reshape((-1, 1)).broadcast_to(x.shape) / self.dim
        diff = x - mean
        vars = ops.summation(diff ** 2, axes=(1,)).reshape((-1, 1)).broadcast_to(x.shape) / self.dim
        norm = diff / (ops.power_scalar(vars + self.eps, 0.5))  
        
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)

        return weight * norm + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            dropout = init.randb(*x.shape, p=1-self.p) / (1 - self.p)
            x = x * dropout
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn.forward(x)
        ### END YOUR SOLUTION



