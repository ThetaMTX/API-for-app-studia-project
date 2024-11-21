from torchview import draw_graph
from model import SimpleCNN
import torch


##model = SimpleCNN(2)
##print(model)


model_graph = draw_graph(SimpleCNN(2),
                        input_size=(1,3,128,128),
                        expand_nested=False,
                        roll=True,
                        graph_dir = 'TB',
                        hide_inner_tensors = True,
                        hide_module_functions = True,

)
model_graph.visual_graph.render("graph.dot", format="png")
