from models.ofa.imagenet_classification.elastic_nn.networks import OFAResNets


def build_ofa_backbone():
    ofa_model = OFAResNets(
        dropout_rate=0,
        depth_list=[0, 1, 2],
        expand_ratio_list=[0.2, 0.25, 0.35],
        width_mult_list=[0.65, 0.8, 1.0],
    )

    # Set the network to the full version
    ofa_model.set_max_net()
    ofa_model.num_channels = 2048

    # Set all parameters training to false
    for param in ofa_model.parameters():
        param.requires_grad = False

    return ofa_model
