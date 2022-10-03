import os
import argparse
import numpy as np
import math
import json

def writes_descriptor(title, path, ids):
    data_dict = {"ids": list(ids)}

    with open(os.path.join(path, f"{title}.json"), "w") as file:
        json.dump(data_dict, file, indent=4)

def creates_dataset_descriptors(
            dataset_path: str, descriptors_path: str, subset_ratios: list
        ) -> None:
    patient_ids = os.listdir(dataset_path)
    n_patients = len(patient_ids)

    if (1.0 - sum(subset_ratios)) > 0.01:
        raise ValueError("Sum of Subset Ratios MUST be 1")
    
    # Defines subset sizes
    train_size = int(n_patients * subset_ratios[0])
    validation_size = int(n_patients * subset_ratios[1])
    
    # Shuffles patient ids and select from train, validation and test
    patient_ids = np.array(patient_ids)
    np.random.seed(7)
    np.random.shuffle(patient_ids)
    train_ids = patient_ids[:train_size]
    validation_ids = patient_ids[train_size:train_size + validation_size]
    test_ids = patient_ids[train_size + validation_size:]

    writes_descriptor(
        title="train", path=descriptors_path, ids=train_ids
    )
    writes_descriptor(
        title="validation", path=descriptors_path, ids=validation_ids
    )
    writes_descriptor(
        title="test", path=descriptors_path, ids=test_ids
    )

if __name__== "__main__":
    parser = argparse.ArgumentParser(
        description="Utilitarian script to separate patients' IDs into train," \
                    " validation, and test cases"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to whole dataset"
    )
    parser.add_argument(
        "descriptors_path", type=str, help="Path to save subset descriptors"
    )
    parser.add_argument(
        "-r", "--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
        help="Subset's ratio of the whole dataset " \
                "[Train, Validation and Test Ratios]"
    )
    args = parser.parse_args()

    creates_dataset_descriptors(
        dataset_path=args.dataset_path,
        descriptors_path=args.descriptors_path,
        subset_ratios=args.ratios
    )