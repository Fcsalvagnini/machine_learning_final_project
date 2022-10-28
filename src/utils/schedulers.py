from monai import optimizers as monai_lr_schedulers
from torch.optim import lr_scheduler


def get_scheduler(scheduler_fn, optimizer, from_monai=False, **kwargs):
    """
    Create an learning rate scheduler object from monai or PyTorch libs
    Args:
        scheduler_fn: Scheduler Function Name
        from_monai: Flag indicating the lib )
    Returns:
        Scheduler object
    """
    libname = lr_scheduler
    if from_monai:
        libname = monai_lr_schedulers
    return getattr(libname, scheduler_fn)(optimizer, **kwargs)
