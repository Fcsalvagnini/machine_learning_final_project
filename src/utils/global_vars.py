# Global variables (Dict mappings)
import logging
import torch
from models.building_blocks import Conv3DBlock

LOGGING_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

BUILDING_BLOCKS = {
    "conv_3D_block": Conv3DBlock
}

NORMALIZATIONS = {
    "InstanceNorm3D": torch.nn.InstanceNorm3d
}

ACTIVATIONS = {
    "LeakyReLU": torch.nn.LeakyReLU,
    "Sigmoid": torch.nn.Sigmoid
}