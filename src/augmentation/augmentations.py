from monai import transforms


def get_augmentations(aug_fn, keys, **kwargs):
    """
    Get augmentation objects based on Monai Lib
    Args:
        aug_fn: augmentation function name
        keys: dict with image and label info
    Returns:
        Data augmentation object
    """
    if aug_fn == "ToTensor":
        return getattr(transforms, aug_fn)(**kwargs)
    else:
        return getattr(transforms, aug_fn)(keys=keys, **kwargs)


class AugmentationPipeline:
    def __init__(self, augmentation_dict) -> None:
        self._pipeline = []
        for aug_fn, aug_params in augmentation_dict.items():
            # some augmentations should only be apllied in the image
            apply_on_label = aug_params.pop("apply_on_label")
            aug = get_augmentations(
                aug_fn,
                keys=["image", "label"] if apply_on_label else ["image"],
                **aug_params,
            )
            self._pipeline.append(aug)
        # Ensuring channel in first (Monai specification)
        self._channel_first = transforms.EnsureChannelFirstd(
            keys=["image", "label"]
        )

    def __call__(self, image, label):
        data_dict = {"image": image, "label": label}
        
        # data_dict = self._channel_first(data_dict)
        for aug in self._pipeline:
            data_dict = aug(data_dict)
        return data_dict["image"], data_dict["label"]


# Checking Operation:
if __name__ == "__main__":
    # (../src) python3 -m utils.data_augmentation
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import yaml

    from utils.configurator import AugmentationsConfigs

    with open("experiment_configs/nn_unet_nvidia.yaml") as yaml_file:
        experiment_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    cfg = AugmentationsConfigs(
        experiment_configs["train_configs"]["data_loader"]
    )

    img = (
        nib.load("/data/BraTS2021_00000/00000057_brain_flair.nii")
        .get_fdata()
        .astype(np.float32)
    )
    lbl = (
        nib.load("data/BraTS2021_00000/00000057_final_seg.nii")
        .get_fdata()
        .astype(np.uint8)
    )
    pipeline = AugmentationPipeline(cfg.augmentations)
    transformed_img, transformed_lbl = pipeline(img, lbl)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes[0, 0].imshow(img[:, :, 75])
    axes[0, 1].imshow(lbl[:, :, 75])
    axes[1, 0].imshow(transformed_img[0, :, :, 75])
    axes[1, 1].imshow(transformed_lbl[0, :, :, 75])
    plt.show()
