import os
import numpy as np


class SaveVoxel:


    def save(self, image, name):
        if not (type) is np.ndarray:
            image = image.numpy()
        img_name = "result_images/" + os.path.join(name.replace(".nii.gz", ".npy").split("/")[-1])
        print(img_name)
        np.save(img_name, image, allow_pickle=False)
        return True
