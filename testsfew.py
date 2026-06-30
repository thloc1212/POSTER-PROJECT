import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix
from utils import load_pretrained_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="SFEW root path containing train/ val/ test/",
    )
    parser.add_argument("--baseline", action="store_true", help="use models.baseline_emotion_hyp")
    parser.add_argument("--poster", action="store_true", help="use models.emotion_hyp")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--modeltype", type=str, default="large", help="small or base or large")
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


def _select_model_builder(args):
    if args.baseline and args.poster:
        raise ValueError("Use only one of --baseline or --poster.")
    if args.poster:
        from models.emotion_hyp import pyramid_trans_expr
    else:
        from models.baseline_emotion_hyp import pyramid_trans_expr
    return pyramid_trans_expr


def test():
    args = parse_args()
    pyramid_trans_expr = _select_model_builder(args)

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

    num_classes = len(test_dataset.classes)
    if num_classes != len(train_dataset.classes):
        raise ValueError("Class count mismatch between train and test folders.")

    class_names = test_dataset.classes
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

    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model = load_pretrained_weights(model, state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pre_labels = []
    gt_labels = []

    with torch.no_grad():
        bingo_cnt = 0
        model.eval()

        for imgs, targets in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs, _ = model(imgs)
            _, predicts = torch.max(outputs, 1)

            bingo_cnt += torch.eq(predicts, targets).sum().item()
            pre_labels.extend(predicts.detach().cpu().tolist())
            gt_labels.extend(targets.detach().cpu().tolist())

    acc = np.around(bingo_cnt / float(len(test_dataset)), 4)
    print(f"Test accuracy: {acc:.4f}.")

    cm = confusion_matrix(gt_labels, pre_labels)

    if args.plot_cm:
        plot_confusion_matrix(np.array(cm), class_names, "SFEW", acc)


if __name__ == "__main__":
    test()
