from torchinfo import summary
from torchview import draw_graph
from models.reu_net import REUNet
from models.al_net import ALNet
from models.u_net import UNetDualDecoder

"""
This skript summaries information regarding the architecture of a network.

A model graph of the neural network is generated in the directory 'model_graphs'.
"""

if __name__ == '__main__':

    net = UNetDualDecoder(mode="baseline", in_channels=3, base_channels=32, con_channels=None)
    name = "baseline_unetdual_nocon"

    summary(net.cuda(), (8, 3, 256, 256), depth=8)
    graph = draw_graph(net.cuda(), input_size=(8, 3, 256, 256), expand_nested=True, depth=10,
                       save_graph=True, directory="model_graphs", filename=name
                       )
    graph.visual_graph
    # print(net)
