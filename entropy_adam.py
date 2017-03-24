import math

import numpy

from chainer import cuda
from chainer import optimizer


class EntropyAdam(optimizer.GradientMethod):

    """Entropy-SGD plus Adam.

    See: https://arxiv.org/abs/1611.01838

    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 eta_prime=0.1, gamma=0.01, L=5, sgld_alpha=0.75,
                 thermal_noise=1e-4):
        self.alpha = alpha
        self.beta1 = beta1 ** L
        self.beta2 = beta2 ** L
        self.eps = eps
        self.eta_prime = eta_prime
        self.gamma = gamma
        self.L = L
        self.sgld_alpha = sgld_alpha
        self.thermal_noise = thermal_noise

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = xp.zeros_like(param.data)
            state['v'] = xp.zeros_like(param.data)
            state['mu'] = param.data.copy()
            state['x'] = param.data.copy()

    def update_one_cpu(self, param, state):
        grad = param.grad

        # SGLD iteration
        x, mu = state['x'], state['mu']
        if (self.t - 1) % self.L == 0:
            x[:] = param.data
            mu[:] = param.data
        dx_prime = grad - self.gamma * (x - param.data)
        noise = (numpy.sqrt(self.eta_prime)
                 * self.thermal_noise
                 * numpy.random.normal(size=param.shape))
        param.data += - self.eta_prime * dx_prime + noise
        mu += (1 - self.sgld_alpha) * (param.data - mu)

        if self.t % self.L == 0:
            # Adam iteration
            grad = self.gamma * (x - mu)
            m, v = state['m'], state['v']
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad * grad - v)
            param.data[:] = x - self.lr * m / (numpy.sqrt(v) + self.eps)

    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        noise = xp.random.normal(
            scale=numpy.sqrt(self.eta_prime) * self.thermal_noise,
            size=param.shape,
            dtype=param.data.dtype)
        if (self.t - 1) % self.L == 0:
            state['x'][:] = param.data
            state['mu'][:] = param.data
        cuda.elementwise(
            'T grad, T x, T gamma, T eta_prime, T thermal_noise, T noise, T sgld_alpha',  # NOQA
            'T param, T mu',
            '''T dx_prime = grad - gamma * (x - param);
               param += - eta_prime * dx_prime + noise;
               mu += (1 - sgld_alpha) * (param - mu);
            ''',
            'entropy_adam_sgld')(
                param.grad, state['x'], self.gamma, self.eta_prime,
                self.thermal_noise, noise, self.sgld_alpha,
                param.data, state['mu'])
        if self.t % self.L == 0:
            cuda.elementwise(
                'T x, T gamma, T lr, T one_minus_beta1, T one_minus_beta2, T eps',  # NOQA
                'T param, T m, T v, T mu',
                '''T grad = gamma * (x - mu);
                   m += one_minus_beta1 * (grad - m);
                   v += one_minus_beta2 * (grad * grad - v);
                   param = x - lr * m / (sqrt(v) + eps);
                ''',
                'entropy_adam_adam')(
                    state['x'], self.gamma, self.lr, 1 - self.beta1,
                    1 - self.beta2, self.eps,
                    param.data, state['m'], state['v'], state['mu'])

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** (self.t // self.L)
        fix2 = 1. - self.beta2 ** (self.t // self.L)
        return self.alpha * math.sqrt(fix2) / fix1
