import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

def probabilistic_augmentation(
            probability:float, augmented:torch.Tensor, original:torch.Tensor
        ) -> torch.Tensor:
    condiction = fn.cast(
        fn.random.coin_flip(probability=probability),
        dtype=types.DALIDataType.BOOL
    )
    negative_condition = condiction ^ True

    return condiction * augmented + negative_condition * original