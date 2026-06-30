import warnings

warnings.filterwarnings("ignore")

import argparse
import os
from time import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from data_preprocessing.dataset_ferplus import FERPlusDataset


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
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id list, e.g. 0 or 0,1")
    parser.add_argument("--save_dir", type=str, default="./checkpoint", help="Checkpoint output directory")
    parser.add_argument("--save_prefix", type=str, default=None, help="Checkpoint filename prefix")
    parser.add_argument("--log_every", type=int, default=20, help="Log progress every N iterations")
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


def _default_prefix(dataset_name):
    return f"{dataset_name}_resnet18"


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


def _build_datasets(args):
    train_transform = transforms.Compose(
        [
            transforms.Lambda(_to_rgb_pil),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.1)),
        ]
    )
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
            train_dataset = datasets.ImageFolder(os.path.join(datapath, "train"), transform=train_transform)
            val_dataset = datasets.ImageFolder(os.path.join(datapath, "test"), transform=eval_transform)
            if train_dataset.class_to_idx != val_dataset.class_to_idx:
                raise ValueError("Class mapping mismatch between train and test folders.")
            num_classes = len(train_dataset.classes)
        else:
            train_dataset = RafDataSet(datapath, train=True, transform=train_transform, basic_aug=True)
            val_dataset = RafDataSet(datapath, train=False, transform=eval_transform)
            num_classes = 7
        return train_dataset, val_dataset, num_classes

    if args.dataset == "affectnet":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        train_dataset = Affectdataset(args.datapath, train=True, transform=train_transform, basic_aug=True)
        val_dataset = Affectdataset(args.datapath, train=False, transform=eval_transform)
        return train_dataset, val_dataset, 7

    if args.dataset == "affectnet8class":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        train_dataset = Affectdataset_8class(args.datapath, train=True, transform=train_transform, basic_aug=True)
        val_dataset = Affectdataset_8class(args.datapath, train=False, transform=eval_transform)
        return train_dataset, val_dataset, 8

    if args.dataset == "ferplus":
        if not args.datapath or not os.path.exists(args.datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {args.datapath}")
        train_dataset = FERPlusDataset(args.datapath, train=True, transform=train_transform, basic_aug=True)
        val_dataset = FERPlusDataset(args.datapath, train=False, transform=eval_transform)
        return train_dataset, val_dataset, 8

    if args.dataset == "sfew":
        if (
            not args.datapath
            or not os.path.isdir(os.path.join(args.datapath, "train"))
            or not os.path.isdir(os.path.join(args.datapath, "val"))
            or not os.path.isdir(os.path.join(args.datapath, "test"))
        ):
            raise FileNotFoundError(
                "Cannot find SFEW root with train/ val/ test/. Please set --datapath to the dataset root."
            )
        train_dataset = datasets.ImageFolder(os.path.join(args.datapath, "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.datapath, "val"), transform=eval_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Class mapping mismatch between train and val folders.")
        return train_dataset, val_dataset, len(train_dataset.classes)

    if args.dataset == "fer2013":
        if (
            not args.datapath
            or not os.path.isdir(os.path.join(args.datapath, "train"))
            or not os.path.isdir(os.path.join(args.datapath, "test"))
        ):
            raise FileNotFoundError(
                "Cannot find FER2013 root with train/ and test/. Please set --datapath to the dataset root."
            )
        train_dataset = datasets.ImageFolder(os.path.join(args.datapath, "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.datapath, "test"), transform=eval_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Class mapping mismatch between train and test folders.")
        return train_dataset, val_dataset, len(train_dataset.classes)

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def run_training():
    args = parse_args()
    torch.manual_seed(123)
    np.random.seed(123)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Work on GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

    os.makedirs(args.save_dir, exist_ok=True)
    save_prefix = args.save_prefix if args.save_prefix else _default_prefix(args.dataset)
    latest_path = os.path.join(args.save_dir, f"{save_prefix}_latest.pth")
    best_path = os.path.join(args.save_dir, f"{save_prefix}_best.pth")

    train_dataset, val_dataset, num_classes = _build_datasets(args)

    print(f"Using dataset: {args.dataset}")
    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    if args.checkpoint:
        print("Loading checkpoint...", args.checkpoint)
        _load_checkpoint(model, args.checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        seen_samples = 0
        start_time = time()

        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad(set_to_none=True)

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            seen_samples += targets.size(0)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_sum += torch.eq(predicts, targets).sum().item()

            if (batch_i + 1) % args.log_every == 0 or (batch_i + 1) == len(train_loader):
                print(
                    "[Epoch %d/%d][Train %d/%d] loss: %.4f acc: %.4f lr: %.6f"
                    % (
                        epoch,
                        args.epochs,
                        batch_i + 1,
                        len(train_loader),
                        train_loss / iter_cnt,
                        correct_sum / float(seen_samples),
                        optimizer.param_groups[0]["lr"],
                    ),
                    flush=True,
                )

        train_acc = correct_sum / float(len(train_dataset))
        train_loss = train_loss / max(iter_cnt, 1)
        elapsed = (time() - start_time) / 60
        print(
            "[Epoch %d] Train time: %.2f min, Training accuracy: %.4f, Loss: %.4f"
            % (epoch, elapsed, train_acc, train_loss)
        )

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            val_iter = 0
            bingo_cnt = 0
            model.eval()

            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_iter += 1

                _, predicts = torch.max(outputs, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().item()
                pre_labels.extend(predicts.detach().cpu().tolist())
                gt_labels.extend(targets.detach().cpu().tolist())

            val_loss = val_loss / max(val_iter, 1)
            val_acc = bingo_cnt / float(len(val_dataset))
            val_f1 = f1_score(np.array(gt_labels), np.array(pre_labels), average="weighted")

            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1:%.4f" % (epoch, val_acc, val_loss, val_f1))

            if val_acc > best_acc:
                best_acc = val_acc
                print("Saving best checkpoint to", best_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": val_acc,
                    },
                    best_path,
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": val_acc,
                },
                latest_path,
            )

    print("Training finished. Best validation accuracy:", round(best_acc, 4))


if __name__ == "__main__":
    run_training()