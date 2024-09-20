import torch
from torch import nn
import timm
from IPython import embed
from timm.models.layers import DropPath

def forward_attn(self, prompt, x):
    # prompt: num, 768
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    _, prompt_len, _ = prompt.shape
    prompt_q = self.qkv(prompt).reshape(B, prompt_len, 3, self.num_heads, C //
                                        self.num_heads).permute(2, 0, 3, 1, 4)[0]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    prompt_attn = (prompt_q @ k.transpose(-2, -1)) * self.scale
    prompt_attn = prompt_attn.softmax(dim=-1)
    prompt_attn = self.attn_drop(prompt_attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    prompt = (prompt_attn @ v).transpose(1, 2).reshape(B, prompt_len, C)
    prompt = self.proj(prompt)
    prompt = self.proj_drop(prompt)
    return prompt, x


def forward_block(self, prompt, x):
    h_prompt, h_x = self.attn(self.norm1(prompt), self.norm1(x))
    h_prompt = self.drop_path(h_prompt)
    h_x = self.drop_path(h_x)
    x = x + h_x
    prompt = prompt+h_prompt
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    hh_prompt = self.drop_path(self.mlp(self.norm2(prompt)))
    prompt = prompt + hh_prompt
    return prompt, x, h_prompt, hh_prompt


def forward_features(self, x):
    x = self.patch_embed(x)
    # stole cls_tokens impl from Phil Wang, thanks
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = self.pos_drop(x + self.pos_embed)


    q_feat = []
    h_feat = []
    hh_feat = []
    for i in range(12):
        bs = x.shape[0]

        prompt = self.synqt.q[i].unsqueeze(0).expand(bs, -1, -1)  # bs, num, 768
        if i == 0:
            prompt = self.synqt.prompt_ffn[i](prompt)
        else:
            prompt = self.synqt.prompt_ffn[i](prompt+q_feat[-1]*self.synqt.feat_scale)
        prompt = self.synqt.qsm_block[i](prompt)
        prompt = self.synqt.prompt_drop(prompt)

        prompt_len = prompt.shape[1]
        prompt, x, h_prompt, hh_prompt = self.blocks[i](prompt, x)
        if prompt_len > 1:
            prompt = prompt.mean(dim=1).view(bs, 1, 768)
            h_prompt = h_prompt.mean(dim=1).view(bs, 1, 768)
            hh_prompt = hh_prompt.mean(dim=1).view(bs, 1, 768)
        q_feat.append(prompt)
        h_feat.append(h_prompt)
        hh_feat.append(hh_prompt)
    q_feat = torch.cat([q_feat[-1]] + q_feat + h_feat + hh_feat, dim=1)
    q_feat = self.synqt(q_feat)  # bs 768
    q_feat = self.synqt.head_norm(q_feat)
    if self.dist_token is None:
        return self.pre_logits(q_feat)
    else:
        return x[:, 0], x[:, 1]



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp1(nn.Module):
    def __init__(self, in_features=768, out_features=768, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self, hidden=64, drop=0.1):
        super().__init__()
        self.mlp = Mlp(768, hidden, 768, nn.GELU, drop)

    def forward(self, x):
        return x + self.mlp(x)

class QSMBlock(nn.Module):
    def __init__(self, attn_scale=0.1, ffn_scale=0.1, drop=0.1):
        super().__init__()
        self.attn = QSMAttention(attn_scale,drop=drop)
        self.mlp = Adapter(48, 0.1)
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.attn_scale = nn.Parameter(torch.ones(1) * attn_scale)
        self.ffn_scale = nn.Parameter(torch.ones(1) * ffn_scale)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.ffn_scale
        return x

class QSMAttention(nn.Module):
    def __init__(self, attn_scale, drop=0.1):
        super().__init__()
        self.scale = 768 ** -0.5
        self.q_proj = Adapter(8, 0)
        self.k_proj = Adapter(8, 0)
        self.v_proj = Adapter(8, 0)
        self.proj = Adapter(8, 0)
        self.feat_drop = nn.Dropout(drop)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        
        self.q_norm = nn.LayerNorm(768)
        self.kv_norm = nn.LayerNorm(768)
        self.token_mask_ratio = drop

    def forward(self, prompt):
        prompt_len = prompt.shape[1]
        norm_prompt = self.q_norm(prompt)
        norm_ref_feat = self.kv_norm(prompt)
        q = self.q_proj(norm_prompt)
        k = self.k_proj(norm_ref_feat)
        v = self.v_proj(norm_ref_feat)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)

        prompt = x[:, :prompt_len]
        return prompt

class ConditionCombineBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.get_weight = Mlp1(768, self.dim-1)
        self.act = nn.Tanh()

        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, feat):
        # bs, layers, 768
        bs, layers = feat.shape[:2]
        weight = self.get_weight(feat[:, 0]).view(bs, -1)
        weight = self.act(weight)
        cls_weight = torch.ones(bs, 1).cuda()
        weight = torch.cat([cls_weight, weight], dim=1)
        return torch.mean(weight.view(bs, layers, 1)*feat, dim=1)


class ScaleShiftBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.scale, mean=1, std=.02)
        nn.init.normal_(self.shift, std=.02)

    def forward(self, feat):
        feat = feat * self.scale + self.shift
        return feat

class SynQT(nn.Module):
    def __init__(self, args, num=16):
        super().__init__()

        self.q = nn.Parameter(torch.zeros(12, args.token_nums, 768))
        nn.init.uniform_(self.q, -1, 1)
        self.prompt_drop = nn.Dropout(0.1)

        self.pre_norm = nn.Sequential(
            *[nn.LayerNorm(768) for _ in range(12)])
        self.qsm_block = nn.Sequential(
            *[QSMBlock(args.attn_scale, args.ffn_scale) for _ in range(12)])

        self.prompt_ffn = nn.Sequential(
            *[Adapter(48, 0.1) for _ in range(12)])
        
        self.feat_scale = nn.Parameter(torch.ones(1) * args.feat_scale)

        self.out_dim = 37
        self.mlp = Adapter(48, 0.1)
        self.out_combine = ConditionCombineBlock(dim=self.out_dim)
        self.out_norm = nn.LayerNorm(768)

        self.attn_scale = args.attn_scale

        self.head_norm = nn.LayerNorm(768)
        self.drop_path = DropPath(drop_prob=0.3)

    def forward(self, feat):
        bs, layers = feat.shape[:2]
        h = []
        for i in range(layers):
            if i == 0:
                h.append(feat[:,i]+self.mlp(self.out_norm(feat[:, i])))
            else:
                h.append(self.drop_path(feat[:,i]+self.mlp(self.out_norm(feat[:, i]))))
        feat = torch.stack(h, dim=1)
        return self.out_combine(feat)


def forward_vit(self, x):
    x = self.forward_features(x)
    if self.head_dist is not None:
        x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        if self.training and not torch.jit.is_scripting():
            # during inference, return the average of both classifier predictions
            return x, x_dist
        else:
            return (x + x_dist) / 2
    else:
        x = self.head(x)
    return x


def set_synqt(args, model):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.synqt = SynQT(args)
        with torch.no_grad():
            model.synqt.head_norm.weight.set_(model.norm.weight.detach())
            model.synqt.head_norm.bias.set_(model.norm.bias.detach())
        bound_method = forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
        bound_method_forward = forward_vit.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method_forward)
    if type(model) == SynQT:
        return
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Block:
            bound_method = forward_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
            set_synqt(args, _)
        elif type(_) == timm.models.vision_transformer.Attention:
            bound_method = forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
            set_synqt(args, _)
        elif len(list(_.children())) != 0:
            set_synqt(args, _)


def forward_ref_features(self, x):
    x = self.patch_embed(x)
    # stole cls_tokens impl from Phil Wang, thanks
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    ref_features = []
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(
            x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    for i in range(len(self.blocks)):
        x = self.blocks[i](x)
        if self.token_mode == "mean":
            ref_features.append(x.mean(dim=1))  # bs, 768
        elif self.token_mode == "cls":
            ref_features.append(x[:, 0])
        else:
            raise RuntimeError("Unknown token type")
    ref_features = torch.stack(ref_features, dim=1)  # bs, 12, 768
    return ref_features


def set_ref(args, model):
    bound_method = forward_ref_features.__get__(model, model.__class__)
    setattr(model, 'forward', bound_method)
