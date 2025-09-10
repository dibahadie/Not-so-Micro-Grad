# Not-so-Micro-Grad

**A not-so-simple autograd engine** inspired by [micrograd](https://github.com/karpathy/micrograd), extended with additional features, neural network utilities, and visualization tools.

---

## ✨ Features

- **Autograd Engine**: Core implementation of `Value` objects that support automatic differentiation.  
- **Tensor Support**: A `Tensor` class for multi-dimensional operations.  
- **Neural Components**:  
  - Layers (`nn/Layers`)  
  - Models (`nn/Models`)  
  - `Neuron` and `Tensor` modules for building and training neural networks.  
- **Visualization**: Tools to visualize computation graphs (`utils/visualize_value.py`).  
- **Examples**: Jupyter notebooks showing how to use the library (`examples/example.ipynb`).  

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Not-so-Micro-Grad.git
cd Not-so-Micro-Grad-main
pip install -e .
```

Requirements:
- Python ≥ 3.12  
- NumPy  
- Matplotlib  

---

## 📖 Usage

### Example: Creating and training a small network

```python
from nn.Value import Value
from nn.Neuron import Neuron

# Simple forward pass
x1, x2 = Value(2.0), Value(-3.0)
n = Neuron(2)          # A neuron with 2 inputs
out = n([x1, x2])      # Forward pass
out.backward()         # Backpropagation
print(out.data, x1.grad, x2.grad)
```

See [examples/example.ipynb](examples/example.ipynb) for a full demo.

---

## 📊 Visualization

You can visualize computation graphs:

```python
from utils.visualize_value import draw_dot
from nn.Value import Value

a, b = Value(2.0), Value(-3.0)
c = a * b
dot = draw_dot(c)
dot.render("graph", format="png")
```

---

## 👩‍💻 Author

- **Diba Hadi Esfangereh**  
  📧 [diba.hadie@gmail.com](mailto:diba.hadie@gmail.com)

---

## 📜 License

This project is licensed under the MIT License — feel free to use and modify.