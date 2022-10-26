
from ast import Call
import nibabel as nib 
from typing import Sequence, Optional, Callable
from torch.utils.data import Dataset, DataLoader

class BrainDataset(Dataset):
    def __init__(self, data: Sequence, transforms: Optional[Callable] = None) -> None:
        pass

    def __getitem__(self, idx) -> Sequence:
        pass

    def __len__(self) -> int:
        pass
        