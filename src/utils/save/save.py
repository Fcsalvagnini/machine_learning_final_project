import os
import numpy as np
import matplotlib.pyplot as plt

class SaveVoxel:


    def save(self, image, name):
        if not (type) is np.ndarray:
            image = image.numpy()
        img_name = "result_images/" + os.path.join(name.replace(".nii.gz", ".npy").split("/")[-1])
        print(img_name)
        np.save(img_name, image, allow_pickle=False)
        return True


class SaveImage:
    def __init__(self):
        pacients = ["01172", "01197"]
        chosen_pacient = pacients[0]
        self.flair = f"BraTS2021_{chosen_pacient}_flair.npy"
        self.t1ce = f"BraTS2021_{chosen_pacient}_t1ce.npy"
        self.seg = f"BraTS2021_{chosen_pacient}_seg.npy"

        self.flair_path = os.path.join("result_images", self.flair)
        self.t1ce_path = os.path.join("result_images", self.t1ce)
        self.seg_path = os.path.join("result_images", self.seg)

        self.im_flair = self.load_images(self.flair_path)
        self.im_t1ce = self.load_images(self.t1ce_path)
        self.im_seg = self.load_images(self.seg_path)

        self.dest_flair_path = os.path.join("images", self.flair.replace("npy", "png"))
        self.dest_t1ce_path = os.path.join("images", self.t1ce.replace("npy", "png"))
        self.dest_seg_path = os.path.join("images", self.seg.replace("npy", "png"))

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        im_slice = 80
        ax[0].imshow(self.im_flair[:, :, im_slice])
        ax[1].imshow(self.im_t1ce[:, :, im_slice])
        ax[2].imshow(self.im_seg[:, :, im_slice])
        plt.show()

    def load_images(self, npy_img):
        return np.load(npy_img)


    def get_array_view(array_view, view, layer):
        if view == "axial":
            array_view = array_view[layer, :, :]
        elif view == "coronal":
            array_view = array_view[:, layer, :]
        elif view == "sagittal":
            array_view = array_view[:, :, layer]

if __name__ == "__main__":
    SaveImage()