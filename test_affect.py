import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class

from utils import *
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet', help='dataset')
    parser.add_argument('--datapath', type=str, default=None, help='dataset root path')
    parser.add_argument('--baseline', action='store_true', help='use models.baseline_emotion_hyp')
    parser.add_argument('--poster', action='store_true', help='use models.emotion_hyp')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()


def _pick_existing_path(candidates, fallback):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return fallback


def _filter_invalid_labels(dataset, max_label):
    targets = getattr(dataset, 'target', None)
    if targets is None:
        return dataset
    targets = np.array(targets)
    invalid_idx = np.where(targets > max_label)[0]
    if len(invalid_idx) == 0:
        return dataset
    keep_idx = np.where(targets <= max_label)[0]
    print(f"[WARN] Found {len(invalid_idx)} samples with label>{max_label}. Filtering them out.")
    return torch.utils.data.Subset(dataset, keep_idx.tolist())


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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    if args.dataset == "rafdb":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the RAF-DB root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        num_classes = 7
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the AffectNet root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        print(f"Using AffectNet7 path: {datapath}")
        num_classes = 7
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_test)
        test_dataset = _filter_invalid_labels(test_dataset, max_label=6)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        if not args.datapath:
            raise FileNotFoundError("Please set --datapath to the AffectNet 8-class root folder.")
        datapath = args.datapath
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset path does not exist: {datapath}")
        print(f"Using AffectNet8 path: {datapath}")
        num_classes = 8
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')


    print("Loading pretrained weights...", args.checkpoint)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    model = model.cuda()


    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets) in enumerate(test_loader):
            outputs, features = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()


        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")
        cm = confusion_matrix(gt_labels, pre_labels)
        # print(cm)

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        cm = np.array(cm)
        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  #
            plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)

        if args.dataset == "affectnet":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]  #
            plot_confusion_matrix(cm, labels_name, 'AffectNet7', acc)

        if args.dataset == "affectnet8class":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]  #
            # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
            # 7: Contempt,
            plot_confusion_matrix(cm, labels_name, 'AffectNet_8class', acc)




if __name__ == "__main__":                    
    test()

