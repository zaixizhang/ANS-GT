import torch
import math
import torch.nn as nn
from torch.nn import functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, get_score=False):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        attn_bias = self.linear_bias(attn_bias).permute(0, 3, 1, 2)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        if get_score:
            score = x[:, :, 0, :] * torch.norm(v, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        if get_score:
            return x, score.mean(dim=1)
        else:
            return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, get_score=False):
        y = self.self_attention_norm(x)
        if get_score:
            _, score = self.self_attention(y, y, y, attn_bias, get_score=True)
            return score
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GT(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        num_global_node,
        attention_dropout_rate,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        self.graph_token = nn.Embedding(self.num_global_node, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(self.num_global_node, attn_bias_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None, get_score=False):
        attn_bias, x = batched_data.attn_bias, batched_data.x
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        node_feature = self.node_encoder(x)         # [n_graph, n_node, n_hidden]
        if perturb is not None:
            node_feature += perturb

        global_node_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        node_feature = torch.cat([node_feature, global_node_feature], dim=1)

        graph_attn_bias = torch.cat([graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(2).
                                     repeat(n_graph, 1, n_node, 1)], dim=1)
        graph_attn_bias = torch.cat(
            [graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(0).
            repeat(n_graph, n_node+self.num_global_node, 1, 1)], dim=2)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        if get_score:
            for i, enc_layer in enumerate(self.layers):
                if i == self.n_layers-1:
                    score = enc_layer(output, graph_attn_bias, get_score=True)
                else:
                    output = enc_layer(output, graph_attn_bias)
            return score
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        output = self.downstream_out_proj(output[:, 0, :])
        return F.log_softmax(output, dim=1)


