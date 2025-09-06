from engin.value import Value
import random

class Neuron():
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1 ))

    def __call__(self, x):
        # we are calculating w * x + b
        act = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        out = act.tanh()
        return out