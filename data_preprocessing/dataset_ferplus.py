import os
from PIL import Image
import torch.utils.data as data

class FERPlusDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, basic_aug=False):
        self.root = root
        self.transform = transform
        self.train = train
        
        # FERPlus structure: train/test both use the same 8 emotion labels.
        # Folder name → label index (8-class mapping)
        sub_folder = 'train' if train else 'test' 
        self.img_dir = os.path.join(self.root, sub_folder)
        
        self.file_paths = []
        self.targets = []
        
        self.folder_to_label = {
            'neutral': 0,
            'happy': 1, 'happiness': 1,  # alias
            'surprise': 2, 'suprise': 2,  # alias (common typo in dataset)
            'sad': 3, 'sadness': 3,      # alias
            'anger': 4, 'angry': 4,      # alias
            'disgust': 5,
            'fear': 6,
            'contempt': 7
        }

        if os.path.exists(self.img_dir):
            for cls_name in os.listdir(self.img_dir):
                label = self.folder_to_label.get(cls_name.lower(), -1)
                if label == -1:
                    continue  # Ignore folders that do not map to a label

                cls_path = os.path.join(self.img_dir, cls_name)
                if os.path.isdir(cls_path):
                    for img_name in os.listdir(cls_path):
                        self.file_paths.append(os.path.join(cls_path, img_name))
                        self.targets.append(label)
        else:
            print(f"Lỗi: Không tìm thấy thư mục {self.img_dir}")
        
        print(f"FERPlus dataset: mode={'train' if train else 'test'}, "
              f"loaded {len(self.file_paths)} images")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target