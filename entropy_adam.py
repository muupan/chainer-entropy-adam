import math

import numpy

from chainer import cuda
from chainer import optimizer


class EntropyAdam(optimizer.GradientMethod):

    """Entropy-SGD plus Adam.

    See: https://arxiv.org/abs/1611.01838

    """

    def __init__(self, alpha=0.001, beta1=0.5, beta2=0.999, eps=1e-8,
                 eta_prime=0.1, gamma=0.01, L=5, sgld_alpha=0.75,
                 thermal_noise=1e-4):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
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
            state['x_prime'] = param.data.copy()

    def update_one_cpu(self, param, state):
        grad = param.grad

        # SGLD iteration
        x_prime, mu = state['x_prime'], state['mu']
        if (self.t - 1) % self.L == 0:
            x_prime[:] = param.data
            mu[:] = param.data
        dx_prime = grad - self.gamma * (param.data - x_prime)
        noise = (numpy.sqrt(self.eta_prime)
                 * self.thermal_noise
                 * numpy.random.normal(size=param.shape))
        x_prime += - self.eta_prime * dx_prime + noise
        mu += (1 - self.sgld_alpha) * (x_prime - mu)

        if self.t % self.L == 0:
            # Adam iteration
            grad = self.gamma * (param.data - mu)
            m, v = state['m'], state['v']
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad * grad - v)
            param.data -= self.lr * m / (numpy.sqrt(v) + self.eps)

    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        noise = xp.random.normal(
            scale=numpy.sqrt(self.eta_prime) * self.thermal_noise,
            size=param.shape,
            dtype=param.data.dtype)
        if (self.t - 1) % self.L == 0:
            state['x_prime'][:] = param.data
            state['mu'][:] = param.data
        cuda.elementwise(
            'T grad, T param, T gamma, T eta_prime, T thermal_noise, T noise, T sgld_alpha',  # NOQA
            'T x_prime, T mu',
            '''T dx_prime = grad - gamma * (param - x_prime);
               x_prime += - eta_prime * dx_prime + noise;
               mu += (1 - sgld_alpha) * (x_prime - mu);
            ''',
            'entropy_adam_sgld')(
                param.grad, param.data, self.gamma, self.eta_prime,
                self.thermal_noise, noise, self.sgld_alpha,
                state['x_prime'], state['mu'])
        if self.t % self.L == 0:
            cuda.elementwise(
                'T gamma, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
                'T param, T m, T v, T x_prime, T mu',
                '''T grad = gamma * (param - mu);
                   m += one_minus_beta1 * (grad - m);
                   v += one_minus_beta2 * (grad * grad - v);
                   param -= lr * m / (sqrt(v) + eps);
                ''',
                'entropy_adam_adam')(
                    self.gamma, self.lr, 1 - self.beta1, 1 - self.beta2,
                    self.eps, param.data, state['m'], state['v'],
                    state['x_prime'], state['mu'])

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** (self.t // self.L)
        fix2 = 1. - self.beta2 ** (self.t // self.L)
        return self.alpha * math.sqrt(fix2) / fix1
