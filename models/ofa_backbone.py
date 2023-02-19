from models.ofa.imagenet_classification.networks.resnets import ResNets
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets

SUBNET_CONFIG = {
    224:{ # 484.6M MACs
        'd': [0, 1, 0, 0, 2], 
        'e': [0.2, 0.25, 0.2, 0.25, 0.2, 0.35, 0.35, 0.25, 0.25, 0.25, 0.2, 0.25, 0.35, 0.2, 0.2, 0.25, 0.2, 0.35], 
        'w': [0, 2, 1, 1, 1, 2]
        },
    288: { # 710.6M MACs
        'd': [2, 2, 0, 0, 1], 
        'e': [0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.2, 0.35, 0.35, 0.25, 0.25, 0.25, 0.2, 0.25, 0.35, 0.25, 0.35], 
        'w': [2, 2, 2, 2, 2, 2]
        },
    352: { # 1015.9M MACs
        'd': [0, 2, 2, 1, 1], 
        'e': [0.2, 0.35, 0.2, 0.2, 0.35, 0.35, 0.25, 0.35, 0.25, 0.25, 0.2, 0.2, 0.25, 0.2, 0.2, 0.35, 0.25, 0.25], 
        'w': [0, 0, 2, 0, 2, 2]
        },
    416: { # 1454.1M MACs
        'd': [2, 1, 2, 2, 2], 
        'e': [0.25, 0.35, 0.35, 0.2, 0.2, 0.25, 0.2, 0.35, 0.35, 0.2, 0.2, 0.25, 0.25, 0.35, 0.2, 0.35, 0.25, 0.35], 
        'w': [2, 1, 1, 0, 0, 2]
        },
    480: { # 1713.2M MACs
        'd': [2, 1, 2, 1, 2], 
        'e': [0.25, 0.35, 0.2, 0.35, 0.35, 0.35, 0.25, 0.35, 0.35, 0.25, 0.35, 0.35, 0.35, 0.2,0.35, 0.25, 0.2, 0.2], 
        'w': [0, 0, 1, 0, 1, 2]
        },
    544: { # 2006.6M MACs
        'd': [0, 1, 1, 1, 1], 
        'e': [0.25, 0.35, 0.25, 0.25, 0.35, 0.35, 0.2, 0.25, 0.25, 0.35, 0.35, 0.25, 0.35, 0.2,0.2, 0.35, 0.25, 0.2], 
        'w': [0, 2, 0, 1, 2, 2]
        },
    608: { # 2358.4M MACs
        'd': [2, 1, 0, 2, 1], 
        'e': [0.25, 0.35, 0.35, 0.25, 0.25, 0.25, 0.35, 0.35, 0.2, 0.25, 0.35, 0.35, 0.35, 0.2,0.25, 0.35, 0.25, 0.2], 
        'w': [2, 0, 1, 2, 2, 2]
        },
    672: { # 2665.0M MACs
        'd': [2, 2, 2, 0, 1], 
        'e': [0.25, 0.25, 0.35, 0.25, 0.35, 0.2, 0.2, 0.35, 0.25, 0.35, 0.35, 0.2, 0.35, 0.35,0.2, 0.35, 0.25, 0.25], 
        'w': [1, 0, 1, 1, 1, 2]
        }
}

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

def set_active_backbone(ofa_model, input_size):
    if input_size not in SUBNET_CONFIG:
        raise NotImplementedError(f"No OFA subnet available for input size {input_size}")
    
    print(f"Setting OFA subnet to {SUBNET_CONFIG[input_size]}")
    ofa_model.set_active_subnet(**SUBNET_CONFIG[input_size])
    return

def get_active_backbone(ofa_model: OFAResNets, input_size) -> ResNets:
    set_active_backbone(ofa_model, input_size)
    subnet = ofa_model.get_active_subnet(preserve_weight=True)
    return subnet

