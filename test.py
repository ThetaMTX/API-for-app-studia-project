from torchviz import make_dot
from model import SimpleCNN
import torch

##model = SimpleCNN(2)
##print(model)

x = torch.randn(1, 3, 128, 128)

# generate predictions for the sample data
y = SimpleCNN(2)(x)

# generate a model architecture visualization
make_dot(y.mean(),
         params=dict(SimpleCNN(2).named_parameters()),
         show_attrs=True,
         show_saved=True).render("MyPyTorchModel_torchviz", format="png")