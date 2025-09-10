from nn.Layers import Layer
from nn import Tensor
import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(nin=in_channels, nout=out_channels)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize filters as Tensors with requires_grad=True
        self.filters = []
        for i in range(out_channels):
            # Create Tensor with random initialization
            filter_data = np.random.randn(in_channels, self.kernel_size[0], self.kernel_size[1])
            filter_tensor = Tensor(filter_data, label=f'filter_{i}')
            self.filters.append(filter_tensor)
        
        # Also store biases as Tensors
        self.biases = []
        for i in range(out_channels):
            bias_tensor = Tensor(np.zeros(1), label=f'bias_{i}')
            self.biases.append(bias_tensor)
        
        # Combine all parameters
        self.parameters = self.filters + self.biases

    def __call__(self, x):
        # Ensure input is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # x shape: (batch_size, in_channels, height, width) or (in_channels, height, width)
        if len(x.data.shape) == 3:
            x_data = x.data[np.newaxis, :]  # Add batch dimension if missing
        else:
            x_data = x.data
        
        batch_size, in_channels, height, width = x_data.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Apply padding if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x_data, ((0, 0), (0, 0), 
                                     (self.padding[0], self.padding[0]), 
                                     (self.padding[1], self.padding[1])), 
                             mode='constant')
        else:
            x_padded = x_data
        
        # Perform convolution using Tensor operations
        outputs = []
        for batch in range(batch_size):
            batch_outputs = []
            for filter_idx, filter_tensor in enumerate(self.filters):
                filter_output_data = np.zeros((out_height, out_width))
                
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract the patch
                        h_start = i * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch_data = x_padded[batch, :, h_start:h_end, w_start:w_end]
                        
                        # Convert patch to Tensor and perform element-wise multiplication
                        patch_tensor = Tensor(patch_data)
                        product = patch_tensor * filter_tensor  # Element-wise multiplication
                        
                        # Sum all elements (convolution operation) and add bias
                        conv_result = product.data.sum() + self.biases[filter_idx].data
                        filter_output_data[i, j] = conv_result
                
                batch_outputs.append(Tensor(filter_output_data))
            
            # Stack outputs for this batch
            if batch_size == 1:
                outputs.append(batch_outputs[0] if len(self.filters) == 1 else batch_outputs)
            else:
                outputs.append(Tensor(np.stack([t.data for t in batch_outputs], axis=0)))
        
        # Return appropriate format
        if batch_size == 1:
            return outputs[0] if len(outputs) == 1 else outputs
        else:
            return Tensor(np.stack([t.data for t in outputs], axis=0))

    def parameters(self):
        return self.parameters