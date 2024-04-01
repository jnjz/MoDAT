import torch
import torch.nn as nn
from model.utils import build_moe_block

class CrossModalMixer(nn.Module):
    def __init__(self, dim=256, n_heads=8, qkv_bias=False, dropout=0.):
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
        
        # fusion_map = torch.einsum('bchw,bc->bchw', feature_map, x.squeeze())
        return fusion_map
    
class MTALayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout=0.0):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    

    def forward(self, audio_feat, x):
        out1 = self.multihead_attn(audio_feat, x, x)[0]
        out1 = audio_feat + self.dropout1(out1)
        out1 = self.norm1(out1)
        out2 = self.ffn(out1)
        return out2

class MTA(nn. Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, num_layers=3, temporal=False):
        super().__init__()
        self.num_layers = num_layers
        self.temporal = temporal
        
        self.layers = nn.ModuleList([MTALayer(embed_dim, num_heads, hidden_dim)
             for i in range(num_layers)])
        
        self.conv = nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim))
        
    def forward(self, x, audio_feat):
        '''
        input 
        audio_feat: query (10, 1, 256)
        x: key value  [10, 256, 28, 28], [10, 256, 14, 14], [10, 256, 7, 7]
        '''
        N, _, C = audio_feat.shape
        if self.num_layers == 4:
            x.append(self.conv(x[-1]))
        flatten_maps = []
        for feature_map in x:
            if self.temporal:
                flatten_maps.append(feature_map.flatten(2).transpose(1, 2).contiguous().view(2, -1, C))   # (bs, T*h*w, 256)
            else:
                flatten_maps.append(feature_map.flatten(2).transpose(1, 2))   # (bs, h*w, 256)
                
        if self.temporal:
            audio_feat = audio_feat.contiguous().view(2, -1, C) # (bs, T, 256)
        
        outputs = []
        for i,layer in enumerate(self.layers):
            outputs.append(layer(audio_feat, flatten_maps[i]))
        output = torch.cat(outputs, dim=1)
            
#         output = audio_feat
#         for i,layer in enumerate(self.layers):
#             output = layer(output, flatten_maps[i])
        
#         if self.temporal: 
#             output = output.contiguous().view(N, 1, C)
        return output


class CSA(nn. Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024):
        super().__init__()
        # self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True) 
        # self.channel_attn = CrossModalMixer(embed_dim, num_heads, qkv_bias=False)
        self.spatial_attn = build_moe_block(dim = 256, num_experts = 16, hidden_dim = 256 * 4, 
                                            activation = nn.LeakyReLU, 
                                            second_policy_train = 'random',
                                            second_policy_eval = 'random', 
                                            second_threshold_train = 0.2,
                                            second_threshold_eval = 0.2,
                                            capacity_factor_train = 1,
                                            capacity_factor_eval = 2., 
                                            loss_coef = 1e-2,
                                            experts = 'spatial'
                                           )
        self.channel_attn = build_moe_block(dim = 256, num_experts = 16, hidden_dim = 256 * 4, 
                                            activation = nn.LeakyReLU, 
                                            second_policy_train = 'random',
                                            second_policy_eval = 'random', 
                                            second_threshold_train = 0.2,
                                            second_threshold_eval = 0.2,
                                            capacity_factor_train = 1,
                                            capacity_factor_eval = 2., 
                                            loss_coef = 1e-2,
                                            experts = 'channel'
                                           )
        
        self.gate_s = nn.Conv2d(embed_dim * 2, 1, kernel_size=1, bias=True)
        self.gate_c = nn.Conv2d(embed_dim * 2, 1, kernel_size=1, bias=True)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, audio_feat):
        '''
        input:
        x : (bs, c, h, w)
        audio_feat: (bs, 1, c)
        '''
        audio_feat = torch.mean(audio_feat, dim=1)
        audio_feat = torch.unsqueeze(audio_feat, dim=1)
        
        channel_out, loss_c = self.channel_attn(x, audio_feat)   # (bs, c, h ,w)
        spatial_out, loss_s = self.spatial_attn(x, audio_feat)   # (bs, c, h ,w)
        cat_fea = torch.cat([channel_out, spatial_out], dim=1)    # (bs, 2c, h ,w)
        attention_vector_c = self.gate_c(cat_fea)   # (bs, 1, h ,w)
        attention_vector_s = self.gate_s(cat_fea)   # (bs, 1, h ,w)
        
        attention_vector = torch.cat([attention_vector_c, attention_vector_s], dim=1)  # (bs, 2, h ,w)
        attention_vector = self.softmax(attention_vector)   # (bs, 2, h ,w)
        attention_vector_c, attention_vector_s = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :] # (bs, 1, h ,w)
        
        output =  channel_out * attention_vector_c + spatial_out * attention_vector_s
        
        return output, (loss_c+loss_s)/2
    

def build_fusion_block(type, **kwargs):
    if type == 'CrossModalMixer':
        return CrossModalMixer(**kwargs)
    elif type == 'CSA':
        return CSA(**kwargs)
    elif type == 'MTA':
        return MTA(**kwargs)
    else:
        raise ValueError
