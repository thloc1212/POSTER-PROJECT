import warnings

warnings.filterwarnings("ignore")
# from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms, datasets
import os

import torch
import argparse
from torch.cuda.amp import autocast, GradScaler
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from data_preprocessing.dataset_ferplus import FERPlusDataset
from data_preprocessing.sam import SAM


from sklearn.metrics import f1_score, confusion_matrix
from time import time
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset')
    parser.add_argument('--datapath', type=str, default=None, help='dataset root path')
    parser.add_argument('--baseline', action='store_true', help='use models.baseline_emotion_hyp')
    parser.add_argument('--poster', action='store_true', help='use models.emotion_hyp')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--lr', type=float, default=0.00004, help='Initial learning rate for AdamW.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    parser.add_argument('--save_dir', type=str, default='./checkpoint', help='Directory to save checkpoints.')
    parser.add_argument('--save_prefix', type=str, default=None, help='Checkpoint filename prefix. Default: <dataset>_<model>.')
    parser.add_argument('--log_every', type=int, default=1, help='Print progress every N iterations.')
    return parser.parse_args()


def _is_raf_root(path):
    return (
        path is not None
        and os.path.exists(os.path.join(path, 'EmoLabel', 'list_patition_label.txt'))
        and os.path.isdir(os.path.join(path, 'Image', 'aligned'))
    )


def _is_raf_imagefolder_root(path):
    return (
        path is not None
        and os.path.isdir(os.path.join(path, 'train'))
        and os.path.isdir(os.path.join(path, 'test'))
    )


def _resolve_raf_datapath(path):
    if not path:
        raise FileNotFoundError("Please set --datapath to the RAF-DB root folder.")

    candidate_paths = [path, os.path.join(path, 'DATASET')]
    for candidate in candidate_paths:
        if _is_raf_root(candidate):
            return candidate, False
        if _is_raf_imagefolder_root(candidate):
            return candidate, True

    raise FileNotFoundError(
        "Cannot find RAF-DB root. Set --datapath to either: (1) folder with EmoLabel/ and Image/aligned/, or (2) folder with train/ and test/."
    )


def _select_model_builder(args):
    if args.baseline and args.poster:
        raise ValueError("Use only one of --baseline or --poster.")
    if args.poster:
        from models.emotion_hyp import pyramid_trans_expr
    else:
        from models.baseline_emotion_hyp import pyramid_trans_expr
    return pyramid_trans_expr


def _resolve_save_prefix(args):
    if args.save_prefix:
        return args.save_prefix
    model_name = 'poster' if args.poster else 'baseline'
    return f"{args.dataset}_{model_name}"




def run_training():
    args = parse_args()
    pyramid_trans_expr = _select_model_builder(args)
    save_prefix = _resolve_save_prefix(args)
    torch.manual_seed(123)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])
    os.makedirs(args.save_dir, exist_ok=True)
    latest_path = os.path.join(args.save_dir, f'{save_prefix}_latest.pth')
    best_path = os.path.join(args.save_dir, f'{save_prefix}_best.pth')


    data_transforms = transforms.Compose([
      #  transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
      #  transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    if args.dataset == "rafdb":
        datapath, use_imagefolder = _resolve_raf_datapath(args.datapath)
        print(f"Using RAF-DB path: {datapath}")
        num_classes = 7
        if use_imagefolder:
            # KaggleHub RAF-DB format: DATASET/train/<class>, DATASET/test/<class>
            train_dataset = datasets.ImageFolder(os.path.join(datapath, 'train'), transform=data_transforms)
            val_dataset = datasets.ImageFolder(os.path.join(datapath, 'test'), transform=data_transforms_val)
        else:
            train_dataset = RafDataSet(datapath, train=True, transform=data_transforms, basic_aug=True)
            val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the AffectNet root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        num_classes = 7
        train_dataset = Affectdataset(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the AffectNet 8-class root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        num_classes = 8
        train_dataset = Affectdataset_8class(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "ferplus":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the FERPlus root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        num_classes = 8
        
        train_dataset = FERPlusDataset(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = FERPlusDataset(datapath, train=False, transform=data_transforms_val)
        
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')

    val_num = val_dataset.__len__()
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # model = Networks.ResNet18_ARM___RAF()

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)


    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        seen_samples = 0
        start_time = time()
        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad(set_to_none=True)
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            seen_samples += targets.size(0)

            with autocast(enabled=torch.cuda.is_available()):
                outputs, features = model(imgs)
                CE_loss = CE_criterion(outputs, targets)
                lsce_loss = lsce_criterion(outputs, targets)
                loss = 2 * lsce_loss + CE_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            if (batch_i + 1) % args.log_every == 0 or (batch_i + 1) == len(train_loader):
                running_acc = correct_sum.float() / float(seen_samples)
                running_loss = train_loss / iter_cnt
                print(
                    "[Epoch %d/%d][Train %d/%d] loss: %.4f acc: %.4f lr: %.6f" %
                    (i, args.epochs, batch_i + 1, len(train_loader), running_loss, running_acc, optimizer.param_groups[0]["lr"]),
                    flush=True,
                )

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs, features = model(imgs)
                    CE_loss = CE_criterion(outputs, targets)
                    loss = CE_loss

                val_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

                if (batch_i + 1) % args.log_every == 0 or (batch_i + 1) == len(val_loader):
                    print(
                        "[Epoch %d/%d][Val %d/%d] loss: %.4f" %
                        (i, args.epochs, batch_i + 1, len(val_loader), val_loss / iter_cnt),
                        flush=True,
                    )

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(gt_labels, pre_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            i, val_acc, val_loss, f1, total_socre))

            torch.save(
                {
                    'iter': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_acc,
                    'args': vars(args),
                },
                latest_path
            )
            print(f'Saved latest checkpoint to {latest_path}.', flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        'iter': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_acc': best_acc,
                        'args': vars(args),
                    },
                    best_path
                )
                print(f'Saved best checkpoint to {best_path}.', flush=True)
                print("best_acc:" + str(best_acc))


if __name__ == "__main__":
    run_training()