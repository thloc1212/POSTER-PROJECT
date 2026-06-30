import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
from .adaptive_fusion_mechanism import AdaptiveFusionModule



def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model




class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])


        for param in self.face_landback.parameters():
            param.requires_grad = False

        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3
        # Thay thế fusion bằng Adaptive Fusion Module
        # Giải quyết: Spatial Attention Misalignment + Semantic Asymmetry
        self.adaptive_fusion = AdaptiveFusionModule(
            embed_dim=512,
            num_patches=49,
            num_classes=self.num_classes
        )

        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)


    def forward(self, x, return_fusion_info=False):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        # Adaptive Fusion với giải quyết conflict
        y_hat, fusion_info = self.adaptive_fusion(x_ir, x_face, return_intermediate=True)
        
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        # Có thể sử dụng fusion_info cho auxiliary losses hoặc visualization
        # fusion_info.keys(): 'alignment', 'uncertainty', 'conflict', 'gating', 'bridging'
        self.last_fusion_info = fusion_info
        
        if return_fusion_info:
            return out, y_feat, fusion_info

        return out, y_feat


