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
    def __init__(self, model_configs, deep_supervision) -> None:
        super().__init__()

        self.encoder_layers = self.get_layers(model_configs.encoder)
        self.decoder_layers = self.get_layers(model_configs.decoder)
        self.skip_connections = self.get_skip_connections(model_configs.skip_connections)
        self.deep_supervision = deep_supervision

        if len(self.skip_connections) < model_configs.depth:
            diff = model_configs.depth - len(self.skip_connections)
            self.skip_connections += [False] * diff

    def get_skip_connections(self, configs):
        skip_layers = []
        for block_name, block_ops in zip(configs.block_names, configs.block_parameters):
            skip_layers.append(True)

        return skip_layers

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
                layers_by_level[-1]._initialize_weights()
            layers_by_level = nn.ModuleList(layers_by_level)

            layers.extend([layers_by_level])

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_representations = []
        outputs = []

        for encoder_layer in self.encoder_layers:
            for encoder_block in encoder_layer:
                x = encoder_block.forward(x)
            intermediate_representations.append(x)

        for idx, decoder_layer, skip, intermediate_representation in \
        enumerate(zip(
            self.decoder_layers, 
            self.skip_connections[::-1], intermediate_representations[::-1]
        )):
            first_block = True
            for decoder_block in decoder_layer:
                if skip and first_block:
                    x = torch.concat([x, intermediate_representation], dim=1)
                    first_block = False
                x = decoder_block.forward(x)
                if self.deep_supervision and idx >= 4:
                    outputs.append(x)

        if not self.deep_supervision:
            outputs = [x]

        return outputs

def get_model(configs, deep_supervision=False):
    model_configs = ModelConfigs(configs["model"])

    segmentation_model = SegmentationModel(
        model_configs=model_configs,
        deep_supervision=deep_supervision
    )

    return segmentation_model
