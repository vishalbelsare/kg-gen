# Neural Networks Architecture

## Basic Components
- [[Neurons]]: Computational units that process inputs
- [[Weights]]: Parameters adjusted during training
- [[Activation Functions]]: Add non-linearity (ReLU, Sigmoid, Tanh)
- [[Loss Functions]]: Measure prediction error

> The power of neural networks comes from their ability to approximate any function given enough neurons and data

### Common Architectures
| Type | Use Case | Example |
|------|----------|---------|
| CNN | Image Processing | ResNet, VGG |
| RNN | Sequential Data | LSTM, GRU |
| Transformer | NLP, Vision | BERT, GPT |

```python
# Simple neural network in PyTorch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
```

This builds on concepts from [[note_01.md|Computer Science Fundamentals]] and relates to my work on [[note_14.md|Image Classification Project]].

#machinelearning #neuralnetworks #deeplearning #AI