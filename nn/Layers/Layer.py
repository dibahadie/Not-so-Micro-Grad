from nn.Neuron import Neuron

class Layer:
    def __init__(self, nin=None, nout=None):
        self.nin = nin
        self.nout = nout

    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement __call__")
    
    def parameters(self):
        return []