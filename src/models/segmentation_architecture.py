import torch
from torch import nn

from src.utils.configurator import ModelConfigs
from src.utils.global_vars import BUILDING_BLOCKS, NORMALIZATIONS, ACTIVATIONS

"""
Except for Parameter, the classes we discuss in this video are all subclasses
of torch.nn.Module. This is the PyTorch base class meant to encapsulate behaviors
specific to PyTorch Models and their components.

One important behavior of torch.nn.Module is registering parameters. If a
particular Module subclass has learning weights, these weights are expressed
as instances of torch.nn.Parameter. The Parameter class is a subclass of
torch.Tensor, with the special behavior that when they are assigned as
attributes of a Module, they are added to the list of that modules parameters.
These parameters may be accessed through the parameters() method on the
Module class.

For further details, please see:
https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html

One other important feature to note: When we checked the weights of our
layer with lin.weight, it reported itself as a Parameter (which is a
subclass of Tensor), and let us know that it’s tracking gradients with
autograd. This is a default behavior for Parameter that differs from Tensor.
"""

class SegmentationModel(nn.Module):
    def __init__(self, model_configs) -> None:
        super().__init__()

        self.encoder_layers = self.get_layers(model_configs.encoder)
        self.decoder_layers = self.get_layers(model_configs.decoder)
        self.skip_connections = self.get_skip_connections(model_configs.skip_connections)
        self.deep_supervisions = self.get_deep_supervisison(model_configs.decoder)

        if len(self.skip_connections) < model_configs.depth:
            diff = model_configs.depth - len(self.skip_connections)
            self.skip_connections += [False] * diff

    def get_skip_connections(self, configs):
        skip_layers = []
        for block_name, block_ops in zip(configs.block_names, configs.block_parameters):
            skip_layers.append(True)

        return skip_layers
        
    def get_deep_supervisison(self, configs):
        deep_supervisions = nn.ModuleList([])
        for block_name, block_ops in zip(configs.block_names, configs.block_parameters):
            block_layer = BUILDING_BLOCKS["_".join(block_name.split("_")[:-1])]
            if "deep_supervision_1" in block_ops.keys():
                deep_supervisions.append(block_layer(**block_ops["deep_supervision_1"]))
            else:
                deep_supervisions.append(torch.nn.Identity())

        return deep_supervisions

    def get_layers(self, configs):
        # Blocks in the same U-Net level will be saved as as list
        layers = nn.ModuleList([])
        for block_name, block_ops in zip(configs.block_names, configs.block_parameters):
            layers_by_level = []
            block_layer = BUILDING_BLOCKS["_".join(block_name.split("_")[:-1])]

            n_blocks = max(block_ops.keys(), key=lambda value: int(value.split("_")[-1]))
            n_blocks = int(n_blocks.split("_")[-1])

            for n in range(1, n_blocks + 1):
                conv_parameters = block_ops[f"conv_{n}"]
                normalization = None
                activation = None
                upsampling = None
                if f"normalization_{n}" in block_ops.keys():
                    operation_parameters = block_ops[f"normalization_{n}"]
                    op_type = operation_parameters.pop("type")
                    normalization = NORMALIZATIONS[op_type](**operation_parameters)
                if f"activation_{n}" in block_ops.keys():
                    operation_parameters = block_ops[f"activation_{n}"]
                    op_type = operation_parameters.pop("type")
                    activation = ACTIVATIONS[op_type](**operation_parameters)
                if f"upsampling_{n}" in block_ops.keys():
                    operation_parameters = block_ops[f"upsampling_{n}"]
                    upsampling = torch.nn.Upsample(**operation_parameters)

                layers_by_level.append(
                    block_layer(
                        **conv_parameters, normalization=normalization, activation=activation, upsampling=upsampling
                    )
                )
                
            
            layers_by_level = nn.ModuleList(layers_by_level)

            layers.extend([layers_by_level])

        return layers

    def forward(self, x: torch.Tensor, training=False) -> torch.Tensor:
        intermediate_representations = []

        for encoder_layer in self.encoder_layers:
            for encoder_block in encoder_layer:
                x, _ = encoder_block.forward(x)
            intermediate_representations.append(x)

        deep_supervisions_output = []

        for decoder_layer, skip, intermediate_representation, deep_supervision in \
        zip(
            self.decoder_layers, self.skip_connections[::-1], intermediate_representations[::-1], self.deep_supervisions
        ):
            first_block = True
            for decoder_block in decoder_layer:
                if skip and first_block:
                    x = torch.concat([x, intermediate_representation], dim=1)
                    first_block = False
                x, not_upsampled = decoder_block.forward(x)
            if not isinstance(deep_supervision, nn.Identity):
                deep_supervisions_output.append(deep_supervision(not_upsampled)[0])
        #print(x.shape)

        if training:
            return x, deep_supervisions_output
        else:
            return x

def get_model(configs):
    model_configs = ModelConfigs(configs["model"])

    segmentation_model = SegmentationModel(model_configs=model_configs)

    return segmentation_model
