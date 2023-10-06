from torchinfo import summary

from models.reu_net import REUNet

if __name__ == '__main__':
    net = REUNet(mode="graham", net_params={"base_channels": 96, "base_channels_factor": 2, "aspp_inter_channels": 96})
    # net = ALNetLight(mode="yang", net_params={"auxiliary_task": True, "aspp": True, "aspp_inter_channels": 32})
    # net = ALNet(mode="graham", net_params={"auxiliary_task": True})
    # net = ALNetDualDecoder(mode="graham", net_params={"auxiliary_task": True})
    # net = AttentionUNet(in_channels=3, out_channels=1, net_params={
    #         "base_channels": 256,
    #         "bottle_channels": 1024,
    #         "depth": 3,
    #         "aspp": True,
    #         "num_ext_skip": None,
    #         "adapter": None,
    #         "aspp_inter_channels": 16,
    #         "inter_channels_equal_out_channels": False,
    #         "down_mode": "conv",
    #         "min_inter_channels": 256
    #     })
    name = "yang_alnetlight"

    summary(net.cuda(), (8, 3, 256, 256), depth=8)
    # graph = draw_graph(net.cuda(), input_size=(8, 3, 256, 256), expand_nested=True, depth=10,
    #                    save_graph=True, directory="model_graphs", filename=name
    #                    )
    # graph.visual_graph
    # print(net)
