import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="SFEW root path containing train/ val/ test/",
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id list, e.g. 0 or 0,1")
    parser.add_argument("-p", "--plot_cm", action="store_true", help="Plot confusion matrix")
    return parser.parse_args()


def _has_required_splits(path):
    return (
        path is not None
        and os.path.isdir(os.path.join(path, "train"))
        and os.path.isdir(os.path.join(path, "val"))
        and os.path.isdir(os.path.join(path, "test"))
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


def test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Work on GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

    if not _has_required_splits(args.datapath):
        raise FileNotFoundError(
            "Cannot find SFEW root with train/ val/ test/. "
            "Please set --datapath to the dataset root."
        )

    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.datapath, "train"))
    test_dataset = datasets.ImageFolder(os.path.join(args.datapath, "test"), transform=test_transform)

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Class mapping mismatch between train and test folders.")

    class_names = test_dataset.classes
    num_classes = len(class_names)

    print("Using SFEW path:", args.datapath)
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
        plot_confusion_matrix(np.array(cm), class_names, "SFEW_ResNet18", acc)


if __name__ == "__main__":
    test()
