import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op='', expression='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.expression = expression if expression else str(data)
        self.label = label
        self.grad = 0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data}, expr={self.expression}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+', f"({self.expression}+{other.expression})")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*', f"({self.expression}*{other.expression})")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-', f"({self.expression}-{other.expression})")
        
        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        # Handle both Value objects and regular numbers
        if isinstance(other, Value):
            exponent = other.data
            expr = f"({self.expression}**{other.expression})"
        else:
            exponent = other
            other = Value(exponent)
            expr = f"({self.expression}**{exponent})"
        
        out = Value(self.data ** exponent, (self, other), f'**', expr)
        
        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad
            if isinstance(other, Value):
                other.grad += ((self.data ** exponent) * math.log(self.data)) * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh', f"tanh({self.expression})")
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
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
        
        # Go one variable at a time and apply the chain rule
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()