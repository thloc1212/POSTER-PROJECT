import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import random
from torchvision.datasets import DatasetFolder, ImageFolder


CLASS_TO_IDX = {
    'neutral': 0,
    'happy': 1,
    'happiness': 1,
    'sad': 2,
    'sadness': 2,
    'surprise': 3,
    'fear': 4,
    'disgust': 5,
    'anger': 6,
    'contempt': 7,
}

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

class Affectdataset_8class(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, basic_aug=False, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        self.file_paths = []
        self.target = []

        if self._has_legacy_affectnet_layout():
            self._build_from_legacy_annotations()
        else:
            self._build_from_train_test_folders()

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]
        #######################################################################################

    def _has_legacy_affectnet_layout(self):
        if self.train:
            ann_path = os.path.join(self.root, 'train_set/train_annotations_8class.txt')
        else:
            ann_path = os.path.join(self.root, 'valid_set/valid_annotations_8class.txt')
        return os.path.exists(ann_path)

    def _build_from_legacy_annotations(self):
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train = pd.read_csv(
            os.path.join(self.root, 'train_set/train_annotations_8class.txt'),
            sep=' ',
            header=None,
        )
        df_valid = pd.read_csv(
            os.path.join(self.root, 'valid_set/valid_annotations_8class.txt'),
            sep=' ',
            header=None,
        )
        dataset = df_train if self.train else df_valid

        if self.dataidxs is not None:
            file_names = np.array(dataset.iloc[:, NAME_COLUMN].values)[self.dataidxs]
            self.target = np.array(dataset.iloc[:, LABEL_COLUMN].values)[self.dataidxs]
        else:
            file_names = dataset.iloc[:, NAME_COLUMN].values
            self.target = dataset.iloc[:, LABEL_COLUMN].values

        for f in file_names:
            if self.train:
                path = os.path.join(self.root, 'train_set/images', f)
            else:
                path = os.path.join(self.root, 'valid_set/images', f)
            self.file_paths.append(path)

    def _build_from_train_test_folders(self):
        split_candidates = ['Train', 'train'] if self.train else ['Test', 'test', 'Val', 'val', 'Valid', 'valid']
        split_dir = None
        for candidate in split_candidates:
            candidate_path = os.path.join(self.root, candidate)
            if os.path.isdir(candidate_path):
                split_dir = candidate_path
                break

        if split_dir is None:
            raise FileNotFoundError(
                f"Cannot find split folder for {'train' if self.train else 'test/val'} under: {self.root}"
            )

        csv_label_map = self._load_labels_csv_map()
        mismatch_count = 0

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            folder_label = self._label_to_idx(class_name)
            if folder_label is None:
                continue

            for file_name in sorted(os.listdir(class_dir)):
                file_lower = file_name.lower()
                if not file_lower.endswith(IMAGE_EXTS):
                    continue

                file_path = os.path.join(class_dir, file_name)
                rel_path = os.path.relpath(file_path, self.root).replace('\\', '/')
                csv_label = csv_label_map.get(rel_path)
                if csv_label is None:
                    csv_label = csv_label_map.get(file_name)

                if csv_label is None:
                    final_label = folder_label
                else:
                    if csv_label != folder_label:
                        mismatch_count += 1
                    final_label = folder_label

                self.file_paths.append(file_path)
                self.target.append(final_label)

        if self.dataidxs is not None:
            self.file_paths = np.array(self.file_paths)[self.dataidxs].tolist()
            self.target = np.array(self.target)[self.dataidxs]
        else:
            self.target = np.array(self.target)

        if mismatch_count > 0:
            print(f"[AffectNet8] labels.csv mismatch count={mismatch_count}. Using folder labels as source of truth.")

    def _load_labels_csv_map(self):
        csv_path = os.path.join(self.root, 'labels.csv')
        if not os.path.exists(csv_path):
            return {}

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[AffectNet8] Cannot read labels.csv ({e}). Fallback to folder labels.")
            return {}

        if df.empty or len(df.columns) < 2:
            print("[AffectNet8] labels.csv is empty/invalid. Fallback to folder labels.")
            return {}

        path_col, label_col = df.columns[0], df.columns[1]
        label_map = {}
        for _, row in df.iterrows():
            file_key = str(row[path_col]).strip().replace('\\', '/')
            label_value = row[label_col]
            label_idx = self._label_to_idx(label_value)
            if label_idx is None:
                continue
            label_map[file_key] = label_idx
            label_map[os.path.basename(file_key)] = label_idx

        return label_map

    def _label_to_idx(self, label):
        if isinstance(label, (int, np.integer)):
            value = int(label)
            if 0 <= value <= 7:
                return value
            return None

        text = str(label).strip().lower()
        if text.isdigit():
            value = int(text)
            if 0 <= value <= 7:
                return value
            return None

        return CLASS_TO_IDX.get(text)

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.target[idx]
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

# import numpy as np
# import glob
# from os.path import *
#
# ######################Train set  7class: 283901   8class :287651
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/train_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<7:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
# with open('train_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()
#
#
# ############################ Test set    3500(7class) 3999(8class)
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/valid_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<8:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
#
# with open('valid_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()