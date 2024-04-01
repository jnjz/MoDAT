import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch import nn
import torch.nn.functional as F
from model.utils import build_positional_encoding

import math
from inspect import isfunction
from einops import rearrange, pack, unpack

class CrossModalMixer(nn.Module):
    def __init__(self, dim=256, n_heads=8, qkv_bias=False, dropout=0.4):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = (dim // n_heads)**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, feature_map, audio_feature):
        """channel attention for modality fusion

        Args:
            feature_map (Tensor): (bs, L, c)
            audio_feature (Tensor): (bs, 1, c)

        Returns:
            Tensor: (bs, L, c)
        """
        flatten_map = feature_map
        B, N, C = flatten_map.shape

        q = self.q_proj(audio_feature).reshape(
            B, 1, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(flatten_map).reshape(
            B, N, 2, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj_drop(self.proj(x))

        x = x.sigmoid()
        fusion_map = torch.einsum('bnc,bc->bnc', feature_map, x.squeeze())
        return fusion_map
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=256, n_heads=8, bias=False, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, bias=bias, batch_first=batch_first, dropout=0.4)

    def forward(self, feature_map, audio_feature):
        return self.attn(feature_map, audio_feature, audio_feature)[0]
    
# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        offload_unused_experts_to_cpu = True
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        # whether to offload unused experts to cpu, will require optimizer handles conversion of gradients to right device when accumulating
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(self, x, audio_feat):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later
        world_size = 1
        rank = 0

        # the experts in use on the rank
        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        x = rearrange(x, 'b e n d -> e b n d')
        # get the experts in use
        self.all_experts_to_cpu_besides(expert_slice)
        experts = self.experts[expert_slice]

        # route tokens to appropriate experts
        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input, audio_feat)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back
        outs = rearrange(outs, 'e b n d -> b e n d')
        assert outs.shape == shape
        return outs

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = 'channel',
        offload_unused_experts_to_cpu=True,
        positional_encoding=True):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        if experts == 'channel':
            self.experts = Experts(experts = [CrossModalMixer(dim, 8, qkv_bias=False) for _ in range(num_experts)],
                                   offload_unused_experts_to_cpu = offload_unused_experts_to_cpu)
        elif experts == 'spatial':
            self.experts = Experts(experts = [MultiHeadAttention(dim, 8, bias=False, batch_first=True) \
                                              for _ in range(num_experts)], \
                                   offload_unused_experts_to_cpu = offload_unused_experts_to_cpu)
        self.loss_coef = loss_coef
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(type='SinePositionalEncoding', num_feats=128)
        else:
            self.positional_encoding = None

    def forward(self, inputs, audio_feat, **kwargs):
        mask = torch.zeros((inputs.size(0), inputs.size(2), inputs.size(3)), device=inputs.device, dtype=torch.bool)
        pos_emb = self.positional_encoding(mask)
        x = inputs + pos_emb
        
        is_image = x.ndim == 4
        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')     # (b, h*w, c)
            
        b, n, d, e = *x.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(x)
        expert_inputs = einsum('b n d, b n e s -> b e s d', x, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        expert_outputs = self.experts(expert_inputs, audio_feat)  # (bs, num_experts, L//num_experts, C)
        output = einsum('besd, bnes -> bnd', expert_outputs, combine_tensor)
        if is_image:
            output, = unpack(output, ps, 'b * d')
            output = rearrange(output, 'b h w d -> b d h w')
            
        return output, loss * self.loss_coef

def build_moe_block(**kwargs):
    return MoE(**kwargs)