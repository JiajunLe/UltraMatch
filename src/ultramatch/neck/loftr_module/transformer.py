from torch.nn import Module
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(128, 128), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/32 featmap, the max length of 128 corresponds to 4096 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(
            0).float().unsqueeze(-1)  # [H, 1]
        j_position = torch.ones(max_shape).cumsum(
            1).float().unsqueeze(-1)  # [W, 1]

        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = (
            npe[0],
            npe[1],
            npe[2],
            npe[3],
        )
        i_position, j_position = (
            i_position * train_res_H / test_res_H,
            j_position * train_res_W / test_res_W,
        )

        div_term = torch.exp(
            torch.arange(0, d_model // 4, 1).float()
            * (-math.log(10000.0) / (d_model // 4))
        )
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(
            *max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32
        )
        cos = torch.zeros(
            *max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32
        )

        sin[:, :, 0::2] = (
            torch.sin(i_position * div_term).half()
            if ropefp16
            else torch.sin(i_position * div_term)
        )
        sin[:, :, 1::2] = (
            torch.sin(j_position * div_term).half()
            if ropefp16
            else torch.sin(j_position * div_term)
        )
        cos[:, :, 0::2] = (
            torch.cos(i_position * div_term).half()
            if ropefp16
            else torch.cos(i_position * div_term)
        )
        cos[:, :, 1::2] = (
            torch.cos(j_position * div_term).half()
            if ropefp16
            else torch.cos(j_position * div_term)
        )

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer(
            "sin", sin.unsqueeze(0), persistent=False
        )  # [1, H, W, C//2]
        self.register_buffer(
            "cos", cos.unsqueeze(0), persistent=False
        )  # [1, H, W, C//2]

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, : x.size(1), : x.size(2), :]) + (
            self.rotate_half(x) * self.sin[:, : x.size(1), : x.size(2), :]
        )

    def rotate_half(self, x):
        a, b, c, d = x.shape
        x = x.reshape(a, b, c, d // 2, 2)

        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""


def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = (
        x_mask[0].sum(-2)[0],
        x_mask[0].sum(-1)[0],
        source_mask[0].sum(-2)[0],
        source_mask[0].sum(-1)[0],
    )
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0


def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, hw, nhead, dim = m.shape
    m = m.view(bs, mask_h0, mask_w0, nhead, dim)
    # Restore each cropped dimension to the query mask's shape.
    if mask_w0 != x_mask.size(-1):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0),
                    mask_h0,
                    x_mask.size(-1) - mask_w0,
                    nhead,
                    dim,
                    device=m.device,
                    dtype=m.dtype,
                ),
            ],
            dim=2,
        )
    if mask_h0 != x_mask.size(-2):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0),
                    x_mask.size(-2) - mask_h0,
                    x_mask.size(-1),
                    nhead,
                    dim,
                    device=m.device,
                    dtype=m.dtype,
                ),
            ],
            dim=1,
        )
    return m


class Attention(Module):
    def __init__(
        self, nhead=8, dim=256, re=False, implementation="legacy"
    ):
        super().__init__()

        self.nhead = nhead
        self.dim = dim
        if implementation not in {"legacy", "sdpa"}:
            raise ValueError(
                f"Unknown attention implementation: {implementation}"
            )
        self.implementation = implementation
        self.sdpa_query_scale = 20.0 * math.sqrt(self.dim)

    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert (
            q_mask is None and kv_mask is None
        ), "Not support generalized attention mask yet."
        # Scaled Cosine Attention
        # Refer to "Query-key normalization for transformers" and "https://kexue.fm/archives/9859"
        # Keep cosine normalization in FP32 for selective feature AMP, then
        # return to the projected dtype so FP16/BF16 SDPA can use Flash kernels.
        projection_dtype = query.dtype
        query = F.normalize(query.float(), p=2, dim=3).to(projection_dtype)
        key = F.normalize(key.float(), p=2, dim=3).to(projection_dtype)
        if self.implementation == "sdpa":
            # torch 2.0 SDPA always applies 1/sqrt(head_dim). Pre-scaling Q
            # reproduces the original softmax(20 * cosine_similarity).
            query = query.transpose(1, 2) * self.sdpa_query_scale
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            return out.transpose(1, 2)

        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        s = 20.0
        A = torch.softmax(s * QK, dim=2)

        out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(
                query, key, value, q_mask, kv_mask
            )

        query, key, value = map(
            lambda x: rearrange(
                x,
                "n h w (nhead d) -> n (h w) nhead d",
                nhead=self.nhead,
                d=self.dim,
            ),
            [query, key, value],
        )

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)

        return m

    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:
            m = self._forward(query, key, value,
                              q_mask=q_mask, kv_mask=kv_mask)
        else:  # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(
                    self._forward(
                        query[i: i + 1],
                        key[i: i + 1],
                        value[i: i + 1],
                        q_mask=q_mask[i: i + 1],
                        kv_mask=kv_mask[i: i + 1],
                    )
                )
            m = torch.cat(m_list, dim=0)
        return m


class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        agg_size0=2,
        agg_size1=2,
        rope=False,
        npe=None,
        attention_impl="legacy",
        is_self_attention=False,
    ):
        super(AG_RoPE_EncoderLayer, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope
        self.is_self_attention = is_self_attention
        self.qkv_proj = None
        self.kv_proj = None

        # aggregate and position encoding
        self.aggregate = (
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=agg_size0,
                padding=0,
                stride=agg_size0,
                bias=False,
                groups=d_model,
            )
            if self.agg_size0 != 1
            else nn.Identity()
        )
        self.max_pool = (
            torch.nn.MaxPool2d(kernel_size=self.agg_size1,
                               stride=self.agg_size1)
            if self.agg_size1 != 1
            else nn.Identity()
        )
        self.mask_max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.rope_pos_enc = RoPEPositionEncodingSine(
            d_model, max_shape=(128, 128), npe=npe, ropefp16=True
        )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(
            self.nhead, self.dim, implementation=attention_impl
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def fuse_projections_for_inference(self):
        if self.qkv_proj is not None or self.kv_proj is not None:
            return self
        if self.training:
            raise RuntimeError("Projection fusion requires eval mode")

        if (
            self.is_self_attention
            and self.agg_size0 == 1
            and self.agg_size1 == 1
        ):
            self.qkv_proj = nn.Linear(
                self.q_proj.in_features,
                self.q_proj.out_features * 3,
                bias=False,
                device=self.q_proj.weight.device,
                dtype=self.q_proj.weight.dtype,
            )
            with torch.no_grad():
                self.qkv_proj.weight.copy_(
                    torch.cat(
                        (
                            self.q_proj.weight,
                            self.k_proj.weight,
                            self.v_proj.weight,
                        ),
                        dim=0,
                    )
                )
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.kv_proj = nn.Linear(
                self.k_proj.in_features,
                self.k_proj.out_features * 2,
                bias=False,
                device=self.k_proj.weight.device,
                dtype=self.k_proj.weight.dtype,
            )
            with torch.no_grad():
                self.kv_proj.weight.copy_(
                    torch.cat((self.k_proj.weight, self.v_proj.weight), dim=0)
                )
            self.k_proj = self.v_proj = None
        return self

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        query_input = self.aggregate(x).permute(0, 2, 3, 1)
        source_input = self.max_pool(source).permute(0, 2, 3, 1)
        if self.qkv_proj is not None:
            # For agg=1 self attention, query_input and source_input are the
            # same tensor. One LayerNorm and one GEMM replace duplicated norm
            # plus three independent projections.
            query_source = self.norm1(query_input)
            query, key, value = self.qkv_proj(query_source).chunk(3, dim=-1)
        else:
            query = self.norm1(query_input)
            source = self.norm1(source_input)
            query = self.q_proj(query)
            if self.kv_proj is not None:
                key, value = self.kv_proj(source).chunk(2, dim=-1)
            else:
                key, value = self.k_proj(source), self.v_proj(source)
        if x_mask is not None:
            # mask 1/8 to 1/32
            x_mask, source_mask = map(
                lambda x: self.mask_max_pool(
                    self.mask_max_pool(x.float())).bool(),
                [x_mask, source_mask],
            )
        # Positional encoding
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask,
                           kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # Upsample feature
        m = rearrange(
            m, "b (h w) c -> b c h w", h=H0 // self.agg_size0, w=W0 // self.agg_size0
        )  # [N, C, H0, W0]

        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(
                m, size=(H0, W0), mode="bilinear", align_corners=False
            )  # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(
            0, 2, 3, 1))  # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2)  # [N, C, H0, W0]

        return x + m


'''
Modified from EfficientLoFTR
'''
class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        self.agg_size0, self.agg_size1 = config["agg_size0"], config["agg_size1"]
        self.rope = config["rope"]

        self_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            config["agg_size0"],
            config["agg_size1"],
            config["rope"],
            config["npe"],
            config.get("attention_impl", "legacy"),
            is_self_attention=True,
        )
        cross_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            config["agg_size0"],
            config["agg_size1"],
            False,
            config["npe"],
            config.get("attention_impl", "legacy"),
            is_self_attention=False,
        )

        self.layers = nn.ModuleList(
            [
                (
                    copy.deepcopy(self_layer)
                    if _ == "self"
                    else copy.deepcopy(cross_layer)
                )
                for _ in self.layer_names
            ]
        )
        self._reset_parameters()
        self._projections_fused = False

    def fuse_projections_for_inference(self):
        if self._projections_fused:
            return self
        if self.training:
            raise RuntimeError("Projection fusion requires eval mode")
        for layer in self.layers:
            layer.fuse_projections_for_inference()
        self._projections_fused = True
        return self

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == "self":
                can_batch = (
                    self._projections_fused
                    and not self.training
                    and not torch.is_grad_enabled()
                    and feat0.shape[1:] == feat1.shape[1:]
                    and (
                        (mask0 is None and mask1 is None)
                        or (
                            mask0 is not None
                            and mask1 is not None
                            and mask0.shape[1:] == mask1.shape[1:]
                        )
                    )
                )
                if can_batch:
                    feature = torch.cat((feat0, feat1), dim=0)
                    mask = (
                        None
                        if mask0 is None
                        else torch.cat((mask0, mask1), dim=0)
                    )
                    feature = layer(feature, feature, mask, mask)
                    feat0, feat1 = feature.chunk(2, dim=0)
                else:
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
