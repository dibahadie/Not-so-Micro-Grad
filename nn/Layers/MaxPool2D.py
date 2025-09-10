from nn.Layers import Layer
import numpy as np
from nn import Tensor

import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride else kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding
        self.mask = None

    def __call__(self, x):
        batch_size, channels, in_h, in_w = x.data.shape

        # Add padding
        if self.padding > 0:
            padded_data = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), 
                                        (self.padding, self.padding)), mode='constant')
        else:
            padded_data = x.data

        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size[1]) // self.stride[1] + 1

        # Initialize output and mask
        out_data = np.zeros((batch_size, channels, out_h, out_w))
        self.mask = np.zeros_like(padded_data)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        window = padded_data[b, c, h_start:h_end, w_start:w_end]
                        out_data[b, c, i, j] = np.max(window)
                        
                        # Store mask
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[b, c, h_start + max_idx[0], w_start + max_idx[1]] = 1
        
        out = Tensor(out_data, (x,), 'maxpool2d', f"maxpool2d({x.expression})")
        
        def _backward():
            grad_padded = np.zeros_like(padded_data)
            
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * self.stride[0]
                            h_end = h_start + self.kernel_size[0]
                            w_start = j * self.stride[1]
                            w_end = w_start + self.kernel_size[1]
                            
                            window_mask = self.mask[b, c, h_start:h_end, w_start:w_end]
                            grad_padded[b, c, h_start:h_end, w_start:w_end] += (
                                window_mask * out.grad[b, c, i, j]
                            )
            
            # Remove padding
            if self.padding > 0:
                grad_unpadded = grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                grad_unpadded = grad_padded
            
            x.grad += grad_unpadded
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return []