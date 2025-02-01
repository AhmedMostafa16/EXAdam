# EXAdam: The Power of Adaptive Cross-Moments

Official implementation of **EXAdam** optimizer from the paper ["EXAdam: The Power of Adaptive Cross-Moments"](https://arxiv.org/abs/2412.20302). EXAdam enhances the traditional Adam optimizer by incorporating cross-moment adaptivity and other enhancements, leading to improved convergence and generalization performance.

## Key Features

- 📈 Improved convergence compared to traditional adaptive methods
- 🚀 Enhanced generalization performance
- 🔧 Easy integration with existing PyTorch projects
- 💪 Robust performance across various deep learning tasks
- ⚡ Efficient computation with minimal overhead

<!-- ## Installation -->
<!---->
<!-- ```bash -->
<!-- pip install exadam -->
<!-- ``` -->
<!---->

## Dependencies

- Python 3.11+
- [PyTorch](https://pytorch.org)
- [NumPy](https://numpy.org)

## Detailed Usage

### Optimizer Parameters

- `lr` (float, optional): Learning rate (default: 0.001)
- `betas` (tuple, optional): Coefficients for computing running averages (default: (0.9, 0.999))
- `eps` (float, optional): Term for numerical stability (default: 1e-8)
- `weight_decay` (float, optional): Weight decay coefficient (default: 0.0)

### Recommended Hyperparameters

| Task Type | Learning Rate | Beta1 | Beta2 | Weight Decay |
| --------- | ------------- | ----- | ----- | ------------ |
| Vision    | 0.001         | 0.9   | 0.999 | 1e-4         |
| NLP       | 0.0003        | 0.9   | 0.999 | 1e-5         |
| RL        | 0.0005        | 0.9   | 0.999 | 0            |

### Complete Training Example

Below is an example code snippet for training a general model with NLL loss with EXAdam.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from exadam import EXAdam

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    # Move model to device
    model = model.to(device)

    # Initialize EXAdam optimizer
    optimizer = EXAdam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    # Use NLL Loss
    criterion = nn.NLLLoss()

    # Training history
    history = {
        'train_loss': []
    }

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Apply log softmax for NLL loss
            log_probs = F.log_softmax(output, dim=1)

            # Calculate loss
            loss = criterion(log_probs, target)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}')

    return history

if __name__ == "__main__":
    # Create dummy dataset
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    batch_size = 32
    num_samples = 1000

    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleNet(input_dim, hidden_dim, output_dim)

    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=10,
        learning_rate=0.001
    )
```

## Citation

If you find EXAdam useful in your research, please cite:

```bibtex
@article{adly2024exadam,
  title={EXAdam: The Power of Adaptive Cross-Moments},
  author={Adly, Ahmed M},
  journal={arXiv preprint arXiv:2412.20302},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The authors would like to thank the PyTorch team for their excellent framework
- Special thanks to all contributors and the research community for their valuable feedback
