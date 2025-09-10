import numpy as np
from nn.Layers import Layer
from nn import Tensor

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights and biases
        # He initialization for ReLU compatibility
        std = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weights = Tensor(np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * std)
        self.bias = Tensor(np.zeros(out_channels))
        
        # Store parameters
        self._parameters = [self.weights, self.bias]

    def __call__(self, x):
        batch_size, in_channels, height, width = x.data.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Apply padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), 
                                     (self.padding[0], self.padding[0]), 
                                     (self.padding[1], self.padding[1])), 
                             mode='constant')
        else:
            x_padded = x.data
        
        # Initialize output
        out_data = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution using im2col approach (more efficient)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract patch
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Convolve and add bias
                        out_data[b, oc, oh, ow] = np.sum(patch * self.weights.data[oc]) + self.bias.data[oc]
        
        out = Tensor(out_data, (x, self.weights, self.bias), 'conv2d', f"conv2d({x.expression})")
        
        def _backward():
            # Initialize gradients
            weights_grad = np.zeros_like(self.weights.data)
            bias_grad = np.zeros_like(self.bias.data)
            x_grad = np.zeros_like(x_padded)
            
            # Backward pass
            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            h_start = oh * self.stride[0]
                            h_end = h_start + self.kernel_size[0]
                            w_start = ow * self.stride[1]
                            w_end = w_start + self.kernel_size[1]
                            
                            # Gradient for weights
                            patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                            weights_grad[oc] += patch * out.grad[b, oc, oh, ow]
                            
                            # Gradient for bias
                            bias_grad[oc] += out.grad[b, oc, oh, ow]
                            
                            # Gradient for input
                            x_grad[b, :, h_start:h_end, w_start:w_end] += (
                                self.weights.data[oc] * out.grad[b, oc, oh, ow]
                            )
            
            # Remove padding from input gradient
            if self.padding[0] > 0 or self.padding[1] > 0:
                x_grad = x_grad[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            
            # Update gradients
            self.weights.grad += weights_grad
            self.bias.grad += bias_grad
            x.grad += x_grad
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return self._parameters