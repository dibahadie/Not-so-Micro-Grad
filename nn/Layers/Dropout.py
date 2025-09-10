from nn.Layers import Layer
import numpy as np
from nn import Tensor

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True
        self.mask = None
    
    def __call__(self, x):
        if self.training:
            # Create binary mask
            self.mask = (np.random.rand(*x.data.shape) > self.p).astype(float)
            scale = 1.0 / (1.0 - self.p)
            out_data = x.data * self.mask * scale
            
            out = Tensor(out_data, (x,), 'dropout', f"dropout({x.expression})")
            
            def _backward():
                grad = out.grad * self.mask * scale
                # Handle broadcasting if needed
                if x.data.shape != out.grad.shape:
                    grad = np.sum(grad, axis=tuple(range(len(out.grad.shape) - len(x.data.shape))), keepdims=True)
                    grad = np.reshape(grad, x.data.shape)
                x.grad += grad
            
            out._backward = _backward
            return out
        else:
            return x
        
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def parameters(self):
        return []