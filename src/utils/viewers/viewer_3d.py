import matplotlib.pyplot as plt
import SimpleITK as sitk
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
def read_img_sitk(img_path: str):
    return sitk.ReadImage(img_path)

@interact
def generate_3d_image(
    layer = (0, 128),
    modality=["t1", "t2", "t1ce", "flair"],
    view = ["axial", "sagittal", "coronal"],
    patient = (253, 253)
):
    if modality == "t1":
        modal = "t1"
    if modality == "t2":
        modal = "t2"
    if modality == "t1ce":
        modal = "t1ce"
    if modality == "flair":
        modal = "flair"
    else:
        #raise ValueError(f"modality not inside of accepted values: {modality}")
        pass
    data_paths = {
        253: {
            "t1":"../../../datasets/BraTS2021_00253/BraTS2021_00253_t1.nii.gz",
            "t2":"../../../datasets/BraTS2021_00253/BraTS2021_00253_t2.nii.gz",
            "t1ce":"../../../datasets/BraTS2021_00253/BraTS2021_00253_t1ce.nii.gz",
            "flair": "../../../datasets/BraTS2021_00253/BraTS2021_00253_flair.nii.gz"
        }
    }
    image = read_img_sitk(data_paths[patient][modal])
    print(image.GetSize())
    array_view = sitk.GetArrayViewFromImage(image)
    if view == "axial":
        array_view = array_view[layer, :, :]
    elif view == "coronal":
        array_view = array_view[:, layer, :]
    elif view == "sagittal":
        array_view = array_view[:, :, layer]
    else:
        #raise ValueError(f"view not inside of accepted values: {view}")
        pass

    plt.figure(figsize=(10, 5))
    plt.imshow(array_view, cmap="gray")
    plt.show()

if __name__ == "__main__":
    generate_3d_image()