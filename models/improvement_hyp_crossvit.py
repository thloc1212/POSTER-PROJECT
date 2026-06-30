import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
from functools import partial


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention_img(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :self.img_chanel, :]
        x_lm = x[:, self.img_chanel:, :]

        B, N, C = x_img.shape
        kv = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        q = x_lm.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_img = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_img = self.proj(x_img)
        x_img = self.proj_drop(x_img)

        return x_img

class Attention_lm(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :self.img_chanel, :]
        x_lm = x[:, self.img_chanel:, :]

        B, N, C = x_lm.shape
        kv = self.kv(x_lm).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        q = x_img.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_lm = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_lm = self.proj(x_lm)
        x_lm = self.proj_drop(x_lm)

        return x_lm

class Block(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_chanel = in_chans + 1
        self.num_channels = in_chans + q_chanel + 2
        self.attn_img = Attention_img(dim, in_chans = in_chans, q_chanel = q_chanel, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.attn_lm = Attention_lm(dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x_img = x[:,:self.img_chanel, :]
        x_lm = x[:,self.img_chanel:, :]
        x_img = x_img + self.drop_path(self.attn_img(self.norm1(x)))
        x_img = x_img + self.drop_path(self.mlp1(self.norm2(x_img)))

        x_lm = x_lm + self.drop_path(self.attn_lm(self.norm3(x)))
        x_lm = x_lm + self.drop_path(self.mlp2(self.norm4(x_lm)))
        x = torch.cat((x_img, x_lm), dim=1)
        x = self.conv(x)
        return x

class PyramidBlock(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block_l = Block(
                dim=dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_m = Block(
            dim=dim//2, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_s = Block(
            dim=dim//4, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        n_channels = (in_chans+1) + (q_chanel+1)

        self.upsample_m_conv = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.upsample_s_conv = nn.Conv1d(n_channels, n_channels, kernel_size=1)


    def forward(self, x):
        x_l = x[0]
        x_m = x[1]
        x_s = x[2]
        x_l = self.block_l(x_l)
        x_m = self.block_m(x_m)
        x_s = self.block_s(x_s)

        x_s_up = F.interpolate(x_s, size=x_m.shape[-1], mode='nearest')
        x_m = self.upsample_s_conv(x_s_up) + x_m

        x_m_up = F.interpolate(x_m, size=x_l.shape[-1], mode='nearest')
        x_l = x_l + self.upsample_m_conv(x_m_up)
        x = [x_l, x_m, x_s]
        return x


class HyVisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, in_chans=49, q_chanel = 49, num_classes=1000, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None,
                 act_layer=None, weight_init=''):

        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.lm_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans + 1, embed_dim))
        self.lm_pos_embed = nn.Parameter(torch.zeros(1, q_chanel + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

        n_channels = (in_chans+1) + (q_chanel+1)
        self.downsample_m = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=4)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            PyramidBlock(
                dim=embed_dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)



    def forward(self, x, x_lm):
        B = x.shape[0]
        cls_img = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_img, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        cls_lm = self.lm_cls_token.expand(B, -1, -1)
        x_lm = torch.cat((cls_lm, x_lm), dim=1)
        x_lm = self.pos_drop(x_lm + self.lm_pos_embed)

        new_x = torch.cat((x, x_lm), dim=1)

        ###############################
        new_x_l = new_x
        new_x_m = self.downsample_m(new_x)
        new_x_s = self.downsample_s(new_x)
        new_x_in = [new_x_l,new_x_m,new_x_s]
        #############################
        new_x_in = self.blocks(new_x_in)
        new_x_l = new_x_in[0]
        new_x_l = self.norm(new_x_l)
        x_class1 = new_x_l[:,0,:]
        x_class2 = new_x_l[:, self.in_chans+1, :]
        x_fused = self.fuse_proj(torch.cat([x_class1, x_class2], dim=-1))

        return x_fused


class ProductCrossDualAttention(nn.Module):
    """Product-cross dual attention between image and landmark token streams.

    Branch A: landmark query -> image key/value
    Branch B: image query -> landmark key/value
    Product-cross gating: multiplicative interaction between both branches.
    """

    def __init__(self, dim=512, num_heads=8, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.scale = self.head_dim ** -0.5

        self.q_img = nn.Linear(dim, dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)

        self.q_lm = nn.Linear(dim, dim)
        self.k_lm = nn.Linear(dim, dim)
        self.v_lm = nn.Linear(dim, dim)

        self.norm_img = nn.LayerNorm(dim)
        self.norm_lm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_proj = nn.Linear(dim * 2, dim)
        self.out_drop = nn.Dropout(proj_drop)
        self.post_norm = nn.LayerNorm(dim)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _reshape_heads(self, x):
        bsz, n_tokens, _ = x.shape
        return x.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _cross_attn(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        return out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.dim)

    def _align_tokens(self, x_img, x_lm):
        # Keep token counts aligned to allow product interaction.
        n_img = x_img.shape[1]
        n_lm = x_lm.shape[1]
        if n_img == n_lm:
            return x_img, x_lm
        min_n = min(n_img, n_lm)
        return x_img[:, :min_n, :], x_lm[:, :min_n, :]

    def forward(self, x_img, x_lm):
        if x_img.dim() == 2:
            x_img = x_img.unsqueeze(1)
        if x_lm.dim() == 2:
            x_lm = x_lm.unsqueeze(1)

        x_img, x_lm = self._align_tokens(x_img, x_lm)

        x_img_n = self.norm_img(x_img)
        x_lm_n = self.norm_lm(x_lm)

        # Branch A: landmark query -> image key/value.
        q_lm = self._reshape_heads(self.q_lm(x_lm_n))
        k_img = self._reshape_heads(self.k_img(x_img_n))
        v_img = self._reshape_heads(self.v_img(x_img_n))
        lm_to_img = self._cross_attn(q_lm, k_img, v_img)

        # Branch B: image query -> landmark key/value.
        q_img = self._reshape_heads(self.q_img(x_img_n))
        k_lm = self._reshape_heads(self.k_lm(x_lm_n))
        v_lm = self._reshape_heads(self.v_lm(x_lm_n))
        img_to_lm = self._cross_attn(q_img, k_lm, v_lm)

        # Learned gate from concatenated cross-attention features.
        gate = torch.sigmoid(self.gate_proj(torch.cat([lm_to_img, img_to_lm], dim=-1)))
        residual_scale = torch.sigmoid(self.res_scale)
        fused_img = x_img + residual_scale * gate * lm_to_img
        fused_lm = x_lm + residual_scale * (1.0 - gate) * img_to_lm

        fused = torch.cat([fused_img, fused_lm], dim=-1)
        fused = self.out_drop(self.out_proj(fused))
        fused = self.post_norm(fused)

        # Return one global representation for classifier head.
        return fused.mean(dim=1)




