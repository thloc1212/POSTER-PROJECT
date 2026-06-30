import warnings

warnings.filterwarnings("ignore")

import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from data_preprocessing.dataset_ferplus import FERPlusDataset
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix


def _to_rgb_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    if hasattr(img, "convert"):
        return img.convert("RGB")

    return Image.fromarray(np.array(img)).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="rafdb",
        choices=["rafdb", "affectnet", "affectnet8class", "ferplus", "sfew", "fer2013"],
        help="dataset name",
    )
    parser.add_argument("--datapath", type=str, default=None, help="dataset root path")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id list, e.g. 0 or 0,1")
    parser.add_argument("-p", "--plot_cm", action="store_true", help="Plot confusion matrix")
    return parser.parse_args()


def _is_raf_root(path):
    return (
        path is not None
        and os.path.exists(os.path.join(path, "EmoLabel", "list_patition_label.txt"))
        and os.path.isdir(os.path.join(path, "Image", "aligned"))
    )


def _is_raf_imagefolder_root(path):
    return path is not None and os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "test"))


def _resolve_raf_datapath(path):
    if not path:
        raise FileNotFoundError("Please set --datapath to the RAF-DB root folder.")

    candidate_paths = [path, os.path.join(path, "DATASET")]
    for candidate in candidate_paths:
        if _is_raf_root(candidate):
            return candidate, False
        if _is_raf_imagefolder_root(candidate):
            return candidate, True

    raise FileNotFoundError(
        "Cannot find RAF-DB root. Set --datapath to either: (1) folder with EmoLabel/ and Image/aligned/, or (2) folder with train/ and test/."
    )


def _build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if isinstance(state, dict) and state:
            keys = list(state.keys())
            if keys[0].startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            else:
                state = {f"module.{k}": v for k, v in state.items()}
        model.load_state_dict(state, strict=False)


def _build_test_dataset(args):
    eval_transform = transforms.Compose(
        [
            transforms.Lambda(_to_rgb_pil),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dataset == "rafdb":
        datapath, use_imagefolder = _resolve_raf_datapath(args.datapath)
        if use_imagefolder:
            train_dataset = datasets.ImageFolder(os.path.join(datapath, "train"))
            test_dataset = datasets.ImageFolder(os.path.join(datapath, "test"), transform=eval_transform)
            if train_dataset.class_to_idx != test_dataset.class_to_idx:
                raise ValueError("Class mapping mismatch between train and test folders.")
            class_names = test_dataset.classes
            return test_dataset, class_names, "RAF-DB_ResNet18"

        test_dataset = RafDataSet(datapath, train=False, transform=eval_transform)
        class_names = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
        return test_dataset, class_names, "RAF-DB_ResNet18"

    if args.dataset == "affectnet":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        test_dataset = Affectdataset(args.datapath, train=False, transform=eval_transform)
        class_names = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
        return test_dataset, class_names, "AffectNet7_ResNet18"

    if args.dataset == "affectnet8class":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        test_dataset = Affectdataset_8class(args.datapath, train=False, transform=eval_transform)
        class_names = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
        return test_dataset, class_names, "AffectNet8_ResNet18"

    if args.dataset == "ferplus":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        test_dataset = FERPlusDataset(args.datapath, train=False, transform=eval_transform)
        class_names = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]
        return test_dataset, class_names, "FERPlus_ResNet18"

    if args.dataset == "sfew":
        if (
            not args.datapath
            or not os.path.isdir(os.path.join(args.datapath, "train"))
            or not os.path.isdir(os.path.join(args.datapath, "test"))
        ):
            raise FileNotFoundError(
                "Cannot find SFEW root with train/ val/ test/. Please set --datapath to the dataset root."
            )
        train_dataset = datasets.ImageFolder(os.path.join(args.datapath, "train"))
        test_dataset = datasets.ImageFolder(os.path.join(args.datapath, "test"), transform=eval_transform)
        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise ValueError("Class mapping mismatch between train and test folders.")
        return test_dataset, test_dataset.classes, "SFEW_ResNet18"

    if args.dataset == "fer2013":
        if (
            not args.datapath
            or not os.path.isdir(os.path.join(args.datapath, "train"))
            or not os.path.isdir(os.path.join(args.datapath, "test"))
        ):
            raise FileNotFoundError(
                "Cannot find FER2013 root with train/ and test/. Please set --datapath to the dataset root."
            )
        train_dataset = datasets.ImageFolder(os.path.join(args.datapath, "train"))
        test_dataset = datasets.ImageFolder(os.path.join(args.datapath, "test"), transform=eval_transform)
        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise ValueError("Class mapping mismatch between train and test folders.")
        return test_dataset, test_dataset.classes, "FER2013_ResNet18"

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Work on GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

    test_dataset, class_names, cm_name = _build_test_dataset(args)
    num_classes = len(class_names)

    print(f"Using dataset: {args.dataset}")
    print("Classes:", class_names)
    print("Test set size:", len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print("Loading checkpoint...", args.checkpoint)
    _load_checkpoint(model, args.checkpoint, device)

    pre_labels = []
    gt_labels = []

    with torch.no_grad():
        bingo_cnt = 0
        model.eval()

        for imgs, targets in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(imgs)
            _, predicts = torch.max(outputs, 1)

            bingo_cnt += torch.eq(predicts, targets).sum().item()
            pre_labels.extend(predicts.detach().cpu().tolist())
            gt_labels.extend(targets.detach().cpu().tolist())

    acc = np.around(bingo_cnt / float(len(test_dataset)), 4)
    print(f"Test accuracy: {acc:.4f}.")

    cm = confusion_matrix(gt_labels, pre_labels)

    if args.plot_cm:
        plot_confusion_matrix(np.array(cm), class_names, cm_name, acc)


if __name__ == "__main__":
    test()