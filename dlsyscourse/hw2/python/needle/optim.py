"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            key = hash(p)
            if not key in self.u:
                self.u[key] = 0
            grad = self.momentum * self.u[key] + (1 - self.momentum) * (p.grad.detach() + self.weight_decay * p.detach())
            self.u[key] = grad
            p.cached_data -= self.lr * grad.cached_data
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            key = hash(p)
            grad = p.grad.detach() + self.weight_decay * p.detach()
            self.m[key] = self.beta1 * self.m.get(key, 0) + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v.get(key, 0) + (1 - self.beta2) * grad * grad
            u_grad = (self.m[key] / (1 - self.beta1 ** self.t)).detach()
            v_grad = (self.v[key] / (1 - self.beta2 ** self.t)).detach()
            p.cached_data -= self.lr * (u_grad / (v_grad ** 0.5 + self.eps)).cached_data
        ### END YOUR SOLUTION
