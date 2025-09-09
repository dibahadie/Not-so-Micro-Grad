import numpy as np
import math

EPS = 1e-12
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

class Tensor():
    def __init__(self, data: np.ndarray, _children=(), _op='', expression='', label=''):
        self.data = np.array(data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self.expression = expression if expression else str(data)
        self.label = label
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.shape = self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+', f"({self.expression}+{other.expression})")
        
        def _backward():
            # Handle broadcasting if needed
            self_grad = out.grad
            other_grad = out.grad
            
            # Sum over broadcasted dimensions if shapes don't match
            if self.shape != out.grad.shape:
                self_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.shape)
            if other.shape != out.grad.shape:
                other_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(other.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.shape)
                
            self.grad += self_grad
            other.grad += other_grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*', f"({self.expression}*{other.expression})")
        
        def _backward():
            # Handle broadcasting for self
            self_grad = other.data * out.grad
            if self.shape != out.grad.shape:
                self_grad = np.sum(self_grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.shape)
            
            # Handle broadcasting for other
            other_grad = self.data * out.grad
            if other.shape != out.grad.shape:
                other_grad = np.sum(other_grad, axis=tuple(range(len(out.grad.shape) - len(other.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.shape)
                
            self.grad += self_grad
            other.grad += other_grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return self + (-other)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            exponent = other
            other_tensor = Tensor(np.array(other))
            expr = f"({self.expression}**{other})"
        else:
            other_tensor = other if isinstance(other, Tensor) else Tensor(np.array(other))
            exponent = other_tensor.data
            expr = f"({self.expression}**{other_tensor.expression})"
        
        out = Tensor(self.data ** exponent, (self, other_tensor), f'**', expr)
        
        def _backward():
            # Gradient for base: d/dx (x^c) = c * x^(c-1)
            base_grad = exponent * (self.data ** (exponent - 1)) * out.grad
            if self.shape != out.grad.shape:
                base_grad = np.sum(base_grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                base_grad = np.reshape(base_grad, self.shape)
            self.grad += base_grad
            
            # Gradient for exponent: d/dc (x^c) = x^c * log(x)
            if hasattr(other_tensor, 'grad'):
                if np.all(self.data > 0):
                    exp_grad = (self.data ** exponent) * np.log(self.data) * out.grad
                    if other_tensor.shape != out.grad.shape:
                        exp_grad = np.sum(exp_grad, axis=tuple(range(len(out.grad.shape) - len(other_tensor.shape))), keepdims=True)
                        exp_grad = np.reshape(exp_grad, other_tensor.shape)
                    other_tensor.grad += exp_grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        other = Tensor(np.array(other)) if not isinstance(other, Tensor) else other
        return other * (self ** -1)
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(np.array(other)) - self
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh', f"tanh({self.expression})")
        
        def _backward():
            grad = (1 - t**2) * out.grad
            if self.shape != out.grad.shape:
                grad = np.sum(grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                grad = np.reshape(grad, self.shape)
            self.grad += grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp', f"exp({self.expression})")
        
        def _backward():
            grad = out.data * out.grad
            if self.shape != out.grad.shape:
                grad = np.sum(grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                grad = np.reshape(grad, self.shape)
            self.grad += grad
        
        out._backward = _backward
        return out
    
    def log(self):
        x = np.maximum(self.data, EPS)
        out = Tensor(np.log(x), (self,), 'log', f"log({self.expression})")
        
        def _backward():
            grad = (1.0 / np.maximum(self.data, EPS)) * out.grad
            if self.shape != out.grad.shape:
                grad = np.sum(grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                grad = np.reshape(grad, self.shape)
            self.grad += grad
        
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), 'ReLU', f"relu({self.expression})")
        
        def _backward():
            grad = (out.grad * (self.data > 0).astype(float))
            if self.shape != out.grad.shape:
                grad = np.sum(grad, axis=tuple(range(len(out.grad.shape) - len(self.shape))), keepdims=True)
                grad = np.reshape(grad, self.shape)
            self.grad += grad
        
        out._backward = _backward
        return out

    def sigmoid(self):
        return Tensor(1.0) / (Tensor(1.0) + (-self).exp())
    
    def backward(self):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = np.ones_like(self.data)
        
        # Go one variable at a time and apply the chain rule
        for node in reversed(topo):
            node._backward()