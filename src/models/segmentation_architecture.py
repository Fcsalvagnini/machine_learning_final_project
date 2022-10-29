import torch
from torch import nn

from src.utils.configurator import ModelConfigs
from src.utils.global_vars import BUILDING_BLOCKS, NORMALIZATIONS, ACTIVATIONS

class SegmentationModel(nn.Module):
    def __init__(self, model_configs) -> None:
        super().__init__()

        self.encoder_layers = self.get_layers(model_configs.encoder)
        self.decoder_layers = self.get_layers(model_configs.decoder)
        self.skip_connections = self.get_skip_connections(model_configs.skip_connections)

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
        layers = []
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

            layers.append(layers_by_level)

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_representations = []

        for encoder_layer in self.encoder_layers:
            for encoder_block in encoder_layer:
                x = encoder_block.forward(x)
            intermediate_representations.append(x)

        for decoder_layer, skip, intermediate_representation in \
        zip(
            self.decoder_layers, self.skip_connections[::-1], intermediate_representations[::-1]
        ):
            first_block = True
            for decoder_block in decoder_layer:
                if skip and first_block:
                    x = torch.concat([x, intermediate_representation], dim=1)
                    first_block = False
                x = decoder_block.forward(x)

        return x

def get_model(configs):
    model_configs = ModelConfigs(configs["model"])

    segmentation_model = SegmentationModel(model_configs=model_configs)

    return segmentation_model