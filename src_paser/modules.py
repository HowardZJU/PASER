import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
import numpy as np
# import ptwt, pywt


def get_window(window_name, length, requires_grad=False):
    """Get the window function with window name and length

    Args:
        window_name (str): 0: all_one, 1: hann, 2: hamming, 3: kaiser
        length (int): the length of each window
        requires_grad (bool, optional): whether set the window trainable. Defaults to False.

    Returns:
        torch.nn.Parameter: window column
    """
    window_set = {
        0: torch.nn.Parameter(torch.ones(length), requires_grad=requires_grad),
        1: torch.nn.Parameter(torch.hann_window(length), requires_grad=requires_grad),
        2: torch.nn.Parameter(
            torch.hamming_window(length), requires_grad=requires_grad
        ),
        3: torch.nn.Parameter(torch.kaiser_window(length), requires_grad=requires_grad),
    }
    return window_set[window_name]


def get_wavelet(wavelet, order):
    if wavelet == 0:
        wavelet = "db"
    return wavelet + str(order)


ACT2FN = {
    "gelu": lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
    "relu": F.relu,
    "swish": lambda x: x * torch.sigmoid(x),
}


class SynthesisV1(nn.Module):
    """
    MHSA with single head for fair comparison
    """

    def __init__(self, args):
        super().__init__()

        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.key(input_tensor)
        value = self.value(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV11(SynthesisV1):
    """
    MHSA with q=k
    """

    def __init__(self, args):
        super().__init__(args)
        del self.key

    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.query(input_tensor)
        value = self.value(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV12(SynthesisV1):
    """
    MHSA with q=k=v
    """

    def __init__(self, args):
        super().__init__(args)
        del self.key
        del self.value

    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.query(input_tensor)
        value = self.query(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV2(SynthesisV1):
    """
    SAR-P
    """

    def __init__(self, args):
        super().__init__(args)
        self.attn_scores = nn.Linear(args.hidden_size, args.max_seq_length)

    def forward(self, input_tensor, attention_mask):
        value = self.value(input_tensor)
        attention_scores = self.attn_scores(input_tensor)
        attention_scores = attention_scores / math.sqrt(value.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV3(SynthesisV1):
    """
    SAR-O
    """

    def __init__(self, args):
        super().__init__(args)
        self.attn_scores = nn.Parameter(
            torch.randn(
                1, args.max_seq_length, args.max_seq_length, dtype=torch.float32
            )
            * 0.02
        )

    def forward(self, input_tensor, attention_mask):
        value = self.value(input_tensor)
        attention_scores = self.attn_scores + attention_mask.squeeze(1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV4(SynthesisV3):
    """
    SAR-R
    """

    def __init__(self, args):
        super().__init__(args)
        self.attn_scores = nn.Parameter(
            torch.randn(
                1, args.max_seq_length, args.max_seq_length, dtype=torch.float32
            )
            * 0.02,
            requires_grad=False,
        )


class SynthesisV5(SynthesisV1):
    """
    SAR without attention matrix
    """

    def forward(self, input_tensor):
        value = self.value(input_tensor)
        hidden_states = self.dense(value)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SelfAttention(nn.Module):
    """
    Canonical SAR
    """

    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores.masked_fill(
            ~attention_mask, torch.tensor(-1e9)
        )
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = attention_probs.masked_fill(
            ~attention_mask, torch.tensor(0.0)
        )

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FastSelfAttention(SelfAttention):
    """
    Canonical FastSelfAttention
    Following the open source code https://github.com/wuch15/Fastformer/blob/main/Fastformer.ipynb
    """

    def __init__(self, args):
        super().__init__(args)

        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

    def forward(self, input_tensor, attention_mask):
        batch, seq_len, _ = input_tensor.shape
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / (
            self.attention_head_size**0.5
        )  # batch, num_head, seq_len
        query_for_score += attention_mask
        query_weight = nn.Softmax(dim=-1)(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_key_layer)
        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .view(-1, 1, self.num_attention_heads * self.attention_head_size)
        )
        pooled_query_repeat = pooled_query.repeat(
            1, seq_len, 1
        )  # batch_size, num_head, seq_len, head_dim

        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat
        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)
        query_key_score += attention_mask
        query_key_weight = nn.Softmax(dim=-1)(query_key_score).unsqueeze(2)
        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        # query = value
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )
        weighted_value = self.out_dropout(weighted_value)
        hidden_states = self.transform(weighted_value) + mixed_key_layer

        return hidden_states


class NoSharedSelfAttention(SelfAttention):
    """
    SAR-N
    """

    def __init__(self, args):
        super().__init__(args)
        del self.query
        del self.key
        del self.value
        self.query = [
            nn.Linear(args.hidden_size, self.all_head_size).to("cuda:0")
            for i in range(args.max_seq_length)
        ]
        self.key = [
            nn.Linear(args.hidden_size, self.all_head_size).to("cuda:0")
            for i in range(args.max_seq_length)
        ]
        self.value = [
            nn.Linear(args.hidden_size, self.all_head_size).to("cuda:0")
            for i in range(args.max_seq_length)
        ]

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = torch.stack(
            [
                self.query[i](input_tensor[:, i, :])
                for i in range(input_tensor.shape[1])
            ],
            dim=1,
        )
        mixed_key_layer = torch.stack(
            [self.key[i](input_tensor[:, i, :]) for i in range(input_tensor.shape[1])],
            dim=1,
        )
        mixed_value_layer = torch.stack(
            [
                self.value[i](input_tensor[:, i, :])
                for i in range(input_tensor.shape[1])
            ],
            dim=1,
        )

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class OverfitSelfAttention(SelfAttention):
    """
    SAR-N+
    """

    def __init__(self, args):
        super().__init__(args)
        del self.query
        del self.key
        del self.value
        self.query = nn.Linear(
            args.hidden_size * args.max_seq_length,
            self.all_head_size * args.max_seq_length,
        )
        self.key = nn.Linear(
            args.hidden_size * args.max_seq_length,
            self.all_head_size * args.max_seq_length,
        )
        self.value = nn.Linear(
            args.hidden_size * args.max_seq_length,
            self.all_head_size * args.max_seq_length,
        )

    def forward(self, input_tensor, attention_mask):
        batch, seq, hidden = input_tensor.shape
        mixed_query_layer = self.query(
            torch.flatten(input_tensor, start_dim=1)
        ).reshape(batch, seq, hidden)
        mixed_key_layer = self.key(torch.flatten(input_tensor, start_dim=1)).reshape(
            batch, seq, hidden
        )
        mixed_value_layer = self.value(
            torch.flatten(input_tensor, start_dim=1)
        ).reshape(batch, seq, hidden)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_scores = attention_scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class LocalSelfAttention(SelfAttention):
    """
    SAR-Local
    """

    def __init__(self, args):
        super().__init__(args)
        self.local_mask = nn.Parameter(
            (
                torch.triu(
                    torch.ones(args.max_seq_length, args.max_seq_length), args.local
                )
                + torch.tril(
                    torch.ones(args.max_seq_length, args.max_seq_length), -args.local
                )
            )
            * -1e4,
            requires_grad=False,
        )

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        mask = torch.minimum(attention_mask, self.local_mask)
        attention_scores = attention_scores + mask
        attention_scores = attention_scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class PoolingSelfAttention(SelfAttention):
    """
    Canonical PoolingFormer
    """

    def __init__(self, args):
        super().__init__(args)
        self.local_mask = nn.Parameter(
            (
                torch.triu(
                    torch.ones(args.max_seq_length, args.max_seq_length), args.local
                )
                + torch.tril(
                    torch.ones(args.max_seq_length, args.max_seq_length), -args.local
                )
            )
            * -1e4,
            requires_grad=False,
        )
        self.query_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.key_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.value_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.pool = nn.MaxPool1d(kernel_size=args.pool_size, stride=args.pool_size)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        mask = self.local_mask
        attention_scores = attention_scores + mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        output_1 = context_layer.view(*new_context_layer_shape)

        mixed_query_layer = self.query_2(output_1)
        mixed_key_layer = self.pool(self.key_2(output_1).transpose(1, 2)).transpose(
            1, 2
        )
        mixed_value_layer = self.pool(self.value_2(output_1).transpose(1, 2)).transpose(
            1, 2
        )

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = context_layer
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class ConvLayer(nn.Module):
    """
    ConvFormer
    """

    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.conv = nn.Sequential()
        self.padding_len = args.conv_size - 1
        padding_set = {0: "circular", 1: "reflect", 2: "constant"}
        self.padding_mode = padding_set[args.padding_mode]
        self.conv_size = args.conv_size

        if args.conv_name == 0 or args.ablate == 4:
            self.conv.add_module(
                "conv",
                nn.Conv1d(
                    args.hidden_size, args.hidden_size, kernel_size=self.conv_size
                ),
            )
        elif args.conv_name == 1:
            conv = nn.Conv1d(
                args.hidden_size,
                args.hidden_size,
                kernel_size=self.conv_size,
                groups=args.hidden_size,
            )
            if args.initialize == 1:
                init_ratio = 5e-3
                conv.weight.data.normal_(0.0, init_ratio)
                conv.bias.data.normal_(0.0, init_ratio)
            self.conv.add_module("depthwise_conv", conv)
        elif args.conv_name == 2:
            self.conv.add_module(
                "depthwise_conv",
                nn.Conv1d(
                    args.hidden_size,
                    args.hidden_size,
                    kernel_size=self.conv_size,
                    groups=args.hidden_size,
                ),
            )
            self.conv.add_module(
                "pointwise_conv",
                nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=1),
            )

        self.act_fn = ACT2FN[args.hidden_act]
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.ablate = args.ablate

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = input_tensor.transpose(1, 2)
        x = nn.functional.pad(x, (self.padding_len, 0), self.padding_mode)
        x = self.conv(x).transpose(1, 2)
        x = self.act_fn(x) if self.ablate == 5 else x
        hidden_states = self.out_dropout(x)
        if self.ablate == 2:
            hidden_states = self.layernorm(hidden_states)
        elif self.ablate == 3:
            hidden_states = hidden_states + input_tensor
        elif self.ablate == 7:
            hidden_states = hidden_states
        else:
            hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class LineLayer(nn.Module):
    """
    Linear Token-mixer
    """

    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.mlp = nn.Sequential()
        mlp = nn.Linear(args.max_seq_length, args.max_seq_length)
        if args.initialize == 1:
            init_ratio = 5e-3
            mlp.weight.data.normal_(0.0, init_ratio)
            mlp.bias.data.normal_(0.0, init_ratio)
        self.mlp.add_module("mlp", mlp)

        self.act_fn = ACT2FN[args.hidden_act]
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.ablate = args.ablate

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = input_tensor.transpose(1, 2)
        x = self.mlp(x).transpose(1, 2)
        x = self.act_fn(x) if self.ablate == 5 else x
        hidden_states = self.out_dropout(x)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class CanonicalGRULayer(nn.Module):
    """
    The gru layer used in the xbox. The performance is pool if self.use_residual=True
    """

    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        if args.use_bidirect == 1:
            self.gru = nn.GRU(
                input_size=args.hidden_size,
                hidden_size=args.inner_size // 2,
                batch_first=True,
                bidirectional=True,
                num_layers=args.num_rnn_layers,
            )
        else:
            self.gru = nn.GRU(
                input_size=args.hidden_size,
                hidden_size=args.inner_size,
                batch_first=True,
                bidirectional=False,
                num_layers=args.num_rnn_layers,
            )
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.use_layernorm = args.use_layernorm
        self.dense = nn.Linear(args.inner_size, args.hidden_size)
        self.use_residual = args.use_residual

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x, _ = self.gru(input_tensor)
        hidden_states = self.dense(x)
        hidden_states = self.out_dropout(hidden_states)
        if self.use_residual:
            hidden_states = hidden_states + input_tensor
        if self.use_layernorm:
            hidden_states = self.layernorm(hidden_states)

        return hidden_states


class GRULayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        if args.use_bidirect == 1:
            self.gru = nn.GRU(
                input_size=args.hidden_size,
                hidden_size=args.hidden_size // 2,
                batch_first=True,
                bidirectional=True,
                num_layers=args.num_rnn_layers,
            )
        else:
            self.gru = nn.GRU(
                input_size=args.hidden_size,
                hidden_size=args.hidden_size,
                batch_first=True,
                bidirectional=False,
                num_layers=args.num_rnn_layers,
            )
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.use_layernorm = args.use_layernorm

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x, _ = self.gru(input_tensor)
        hidden_states = self.out_dropout(x) + input_tensor
        if self.use_layernorm:
            hidden_states = self.layernorm(hidden_states)

        return hidden_states


def leCunUniform(tensor):
    """
    LeCun Uniform Initializer
    References:
    [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
    [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3.0 / fan_in)
    nn.init.uniform_(
        tensor, -limit, limit
    )  # fills the tensor with values sampled from U(-limit, limit)


class LMUCell(nn.Module):
    """
    LMU Cell (bad performance)
    Parameters:
        input_size (int) :
            Size of the input vector (x_t)
        hidden_size (int) :
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(
        self, input_size, hidden_size, memory_size, theta, learn_a=False, learn_b=False
    ):
        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)

        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters()

    def initParameters(self):
        """Initialize the cell's parameters"""

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        nn.init.constant_(self.e_m, 0)
        # Initialize kernels
        nn.init.xavier_normal_(self.W_x)
        nn.init.xavier_normal_(self.W_h)
        nn.init.xavier_normal_(self.W_m)

    def stateSpaceMatrices(self, memory_size, theta):
        """Returns the discretized state space matrices A and B"""

        Q = np.arange(memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2 * Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))
        B = R * ((-1.0) ** Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(system=(A, B, C, D), dt=1.0, method="zoh")

        return A, B

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, input_size]
            state (tuple):
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        h, m = state

        # Equation (7) of the paper
        u = (
            F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m)
        )  # [batch_size, 1]
        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B)  # [batch_size, memory_size]
        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) + F.linear(h, self.W_h) + F.linear(m, self.W_m)
        )  # [batch_size, hidden_size]

        return h, m


class LMUSingleLayer(nn.Module):
    """
    LMU layer
    Parameters:
        input_size (int) :
            Size of the input vector (x_t)
        hidden_size (int) :
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(
        self, input_size, hidden_size, memory_size, theta, learn_a=False, learn_b=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.cell = LMUCell(
            input_size, hidden_size, memory_size, theta, learn_a, learn_b
        )

    def forward(self, x, state=None):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, seq_len, input_size]
            state (tuple) : (default = None)
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initial state (h_0, m_0)
        if state == None:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            m_0 = torch.zeros(batch_size, self.memory_size)
            if x.is_cuda:
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)  # state is (h_n, m_n) where n = 1...seq_len

        output = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)

        output = torch.stack(output)  # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        return output


class LMULayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.lmu = LMUSingleLayer(
            args.hidden_size,
            args.hidden_size,
            args.memory_size,
            args.max_seq_length,
            learn_a=False,
            learn_b=False,
        )
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.use_layernorm = args.use_layernorm

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = self.lmu(input_tensor, None)
        hidden_states = self.out_dropout(x) + input_tensor
        if self.use_layernorm:
            hidden_states = self.layernorm(hidden_states)

        return hidden_states


class FConvLayer(nn.Module):
    """
    ConvFormer-F
    """

    def __init__(self, args):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.randn(1, args.conv_size, args.hidden_size, dtype=torch.float32) * 0.02
        )
        self.zeros = nn.Parameter(
            torch.zeros(1, args.max_seq_length - args.conv_size, args.hidden_size),
            requires_grad=False,
        )
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        # padding should be conducted on the right, since fconv is equal to the vanilla convolution with flip.
        weight = torch.cat([self.conv_weight, self.zeros], dim=1)
        weight = torch.fft.rfft(weight, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayer(nn.Module):
    """
    Canonical FMLP-Rec
    """

    def __init__(self, args):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(
                1,
                args.max_seq_length // 2 + 1,
                args.hidden_size,
                2,
                dtype=torch.float32,
            )
            * 0.02
        )
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.window = get_window(args.window, args.max_seq_length)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        input_tensor = torch.einsum("ijk, j->ijk", input_tensor, self.window)
        weight = torch.view_as_complex(self.complex_weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class SpecCrossLayer(FilterLayer):
    def __init__(self, args):
        super().__init__(args)
        self.order = args.order
        self.cross = args.cross

        if self.cross == 1:
            self.complex_weight = nn.Parameter(
                torch.randn(
                    self.order,
                    1,
                    args.max_seq_length // 2 + 1,
                    args.hidden_size,
                    2,
                    dtype=torch.float32,
                )
                * args.initialize_ratio
            )
        elif self.cross == 2:
            self.complex_weight = nn.Parameter(
                torch.randn(
                    self.order,
                    1,
                    args.max_seq_length,
                    args.hidden_size // 2 + 1,
                    2,
                    dtype=torch.float32,
                )
                * args.initialize_ratio
            )
        elif self.cross == 3:
            self.complex_weight = nn.Parameter(
                torch.randn(
                    self.order,
                    1,
                    args.max_seq_length,
                    args.hidden_size // 2 + 1,
                    2,
                    dtype=torch.float32,
                )
                * args.initialize_ratio
            )
        self.bias = nn.Parameter(
            torch.randn(self.order, 1, 1, args.hidden_size, dtype=torch.float32)
            * args.initialize_ratio
        )

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.complex_weight)
        if self.cross == 1:
            input_fft = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
            output_fft = torch.fft.irfft(
                input_fft * weight[0], n=seq_len, dim=1, norm="ortho"
            )
            +self.bias[0]
            for i in range(1, self.order):
                output_fft = (
                    torch.fft.irfft(
                        torch.fft.rfft(output_fft, dim=1, norm="ortho")
                        * weight[i]
                        * input_fft,
                        n=seq_len,
                        dim=1,
                        norm="ortho",
                    )
                    + self.bias[i]
                    + input_tensor
                )
        elif self.cross == 2:
            input_fft = torch.fft.rfft(input_tensor, dim=2, norm="ortho")
            output_fft = torch.fft.irfft(
                input_fft * weight[0], n=hidden, dim=2, norm="ortho"
            )
            +self.bias[0]
            for i in range(1, self.order):
                output_fft = (
                    torch.fft.irfft(
                        torch.fft.rfft(output_fft, dim=2, norm="ortho")
                        * weight[i]
                        * input_fft,
                        n=hidden,
                        dim=2,
                        norm="ortho",
                    )
                    + self.bias[i]
                    + input_tensor
                )
        elif self.cross == 3:
            input_fft = torch.fft.rfft2(input_tensor, norm="ortho")
            output_fft = torch.fft.irfft2(input_fft * weight[0], norm="ortho")
            +self.bias[0]
            for i in range(1, self.order):
                output_fft = (
                    torch.fft.irfft2(
                        torch.fft.rfft2(output_fft, norm="ortho")
                        * weight[i]
                        * input_fft,
                        norm="ortho",
                    )
                    + self.bias[i]
                    + input_tensor
                )
        hidden_states = self.out_dropout(output_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayer2D(FilterLayer):
    def __init__(self, args):
        super().__init__(args)
        self.complex_weight = nn.Parameter(
            torch.randn(
                1,
                args.max_seq_length,
                args.hidden_size // 2 + 1,
                2,
                dtype=torch.float32,
            )
            * 0.02
        )

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.complex_weight)

        x = torch.fft.rfft2(input_tensor, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft2(x, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayerM1(FilterLayer):
    """
    Generate gate via the GAP operator along the time axis.
    """

    def __init__(self, args):
        super().__init__(args)
        self.expert_num = args.expert_num
        self.complex_weight = nn.Parameter(
            torch.randn(
                self.expert_num,
                args.max_seq_length // 2 + 1,
                args.hidden_size,
                2,
                dtype=torch.float32,
            )
            * 0.02
        )
        self.mlp = nn.Linear(args.hidden_size, self.expert_num)
        self.pooling_idx = args.pooling
        self.trunc_gate = args.trunc_gate
        if args.pooling == 1:
            self.pool = nn.AvgPool1d(args.max_seq_length)
        elif args.pooling == 2:
            self.pool = nn.MaxPool1d(args.max_seq_length)

    def forward(self, input_tensor):
        _, seq_len, _ = input_tensor.shape
        if self.pooling_idx == 1:
            valid_len = (input_tensor.sum(-1) != 0).sum(-1).unsqueeze(1).unsqueeze(2)
            input_gate = (
                input_tensor / (valid_len + 1e-4) * 50
            )  # sometimes valid_len = 0
        else:
            input_gate = input_tensor

        if self.trunc_gate == 1:  # truncate the gradient to embedding layer
            input_gate = torch.detach(input_gate)
        gate = self.pool(input_gate.transpose(1, 2)).squeeze()
        gate = self.mlp(gate)
        gate = torch.softmax(gate, dim=1)

        weight = torch.einsum("ij, jklm->iklm", gate, self.complex_weight)
        weight = torch.view_as_complex(weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayerM11(FilterLayerM1):
    """
    Generate gate via the GAP operator along the hidden axis.
    """

    def __init__(self, args):
        super().__init__(args)
        self.mlp = nn.Linear(args.max_seq_length, self.expert_num)

    def forward(self, input_tensor):
        _, seq_len, _ = input_tensor.shape
        gate = self.pool(input_tensor).squeeze()
        gate = self.mlp(gate)
        gate = torch.softmax(gate, dim=-1)
        weight = torch.einsum("ij, jklm->iklm", gate, self.complex_weight)
        weight = torch.view_as_complex(weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayerM12(FilterLayerM1):
    """
    Generate gate via the flattened inputs followed by a MLP layer.
    """

    def __init__(self, args):
        super().__init__(args)
        self.mlp = nn.Linear(args.hidden_size * args.max_seq_length, self.expert_num)

    def forward(self, input_tensor):
        _, seq_len, _ = input_tensor.shape
        gate = self.mlp(input_tensor.flatten(1))
        gate = torch.softmax(gate, dim=1)
        weight = torch.einsum("ij, jklm->iklm", gate, self.complex_weight)
        weight = torch.view_as_complex(weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayerM2(FilterLayerM1):
    """
    Generate filter via filter=MLP(x)
    """

    def __init__(self, args):
        super().__init__(args)
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.mlp1 = nn.Linear(args.max_seq_length, (args.max_seq_length // 2 + 1) * 2)
        self.mlp2 = nn.Linear(
            (args.max_seq_length // 2 + 1) * 2, (args.max_seq_length // 2 + 1) * 2
        )

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        weight = self.mlp2(torch.relu(self.mlp1(input_tensor.transpose(1, 2))))
        weight = weight.reshape(batch, hidden, seq_len // 2 + 1, 2)
        weight = torch.view_as_complex(weight).transpose(1, 2)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayerM3(FilterLayerM1):
    """
    Generate gate by pooling; the pool is not global, but stride=kernel_size=seqlen//2
    """

    def __init__(self, args):
        super().__init__(args)
        self.mlp = nn.Linear(args.hidden_size * 2, self.expert_num)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        gate = torch.max_pool1d(
            input_tensor.transpose(1, 2), kernel_size=seq_len // 2, stride=seq_len // 2
        ).flatten(1)
        gate = self.mlp(gate)
        gate = torch.softmax(gate, dim=1)
        weight = torch.einsum("ij, jklm->iklm", gate, self.complex_weight)
        weight = torch.view_as_complex(weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class WaveletLayer(FilterLayer):
    """
    Wavelet filter (bad performance)
    """

    def __init__(self, args):
        super().__init__(args)
        self.level = args.level
        wavelet = get_wavelet(args.wavelet, args.order)
        self.wavelet = pywt.Wavelet("db3")
        coff = ptwt.wavedec(
            torch.Tensor([i for i in range(50)]), self.wavelet, self.level
        )
        self.filter = [
            nn.Parameter(
                torch.randn(
                    1,
                    args.hidden_size,
                    item.shape[1],
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                )
                * 0.02
            )
            for item in coff
        ]

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = input_tensor.transpose(1, 2).contiguous().view(batch * hidden, seq_len)
        coff = ptwt.wavedec(x, self.wavelet, self.level)
        coff = [item.reshape(batch, hidden, -1) for item in coff]
        coff = [coff[i] * self.filter[i] for i in range(len(coff))]
        coff = [item.reshape(batch * hidden, -1) for item in coff]
        sequence_emb_fft = (
            ptwt.waverec(coff, wavelet=self.wavelet)
            .reshape(batch, hidden, seq_len)
            .transpose(1, 2)
        )

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class STFTLayer(FilterLayer):
    """
    STFT filter
    """

    def __init__(self, args):
        super().__init__(args)
        self.n_fft = args.n_fft
        self.hop_length = args.n_hop
        self.normalized = True if args.normalized == 1 else False
        self.onesided = True if args.onesided == 1 else False
        self.center = True if args.center == 1 else False
        coff = torch.stft(
            torch.Tensor([i for i in range(50)]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            normalized=self.normalized,
            center=self.center,
            return_complex=True,
        )
        self.filter = nn.Parameter(
            torch.randn(1, args.hidden_size, coff.shape[0], coff.shape[1], 2) * 0.02
        )  # batch, hidden, freq//2+1, time, 2

    def forward(self, input_tensor, attention_mask=None):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.filter)
        num_time, num_fft = weight.shape[2], weight.shape[3]

        x = input_tensor.transpose(1, 2).reshape(
            batch * hidden, seq_len
        )  # batch*hidden, seq_len
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            normalized=self.normalized,
            center=self.center,
            return_complex=True,
        )  # batch*hidden, time, n_fft
        x = x.reshape(batch, hidden, num_time, num_fft)
        x = x * weight
        x = x.reshape(batch * hidden, num_time, num_fft)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            length=seq_len,
            normalized=self.normalized,
            center=self.center,
            return_complex=False,
        )
        x = x.reshape(batch, hidden, seq_len)
        x = x.transpose(1, 2).contiguous()
        hidden_states = self.out_dropout(x)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class STFTLayerW(STFTLayer):
    """
    Add window function to STFTLayer
    """

    def __init__(self, args):
        super().__init__(args)
        self.window = get_window(args.window, length=self.n_fft)

    def forward(self, input_tensor, attention_mask=None):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.filter)
        num_time, num_fft = weight.shape[2], weight.shape[3]

        x = input_tensor.transpose(1, 2).contiguous().view(batch * hidden, seq_len)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            normalized=self.normalized,
            center=self.center,
            window=self.window,
            return_complex=True,
        )
        x = x.reshape(batch, hidden, num_time, num_fft)
        x = x * weight
        x = x.reshape(batch * hidden, num_time, num_fft)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            length=seq_len,
            normalized=self.normalized,
            center=self.center,
            window=self.window,
            return_complex=False,
        )
        x = x.reshape(batch, hidden, seq_len)
        x = x.transpose(1, 2).contiguous()
        hidden_states = self.out_dropout(x)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class STFTLayerWM(STFTLayer):
    """
    STFT layer with window function and MoE for filter personalization (bad performance)
    """

    def __init__(self, args):
        super().__init__(args)
        coff = torch.stft(
            torch.Tensor([i for i in range(50)]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            normalized=self.normalized,
            center=self.center,
            return_complex=True,
        )
        self.expert_num = args.expert_num
        freqs, intervals = coff.shape[0], coff.shape[1]

        if args.filter_vary == 1:
            self.filter = nn.Parameter(
                torch.randn(self.expert_num, args.hidden_size, freqs, intervals, 2)
                * 0.02
            )  # exp, hidden, freq//2+1, time, 2
        else:
            self.filter = nn.Parameter(
                torch.randn(self.expert_num, args.hidden_size, freqs, 1, 2) * 0.02
            )  # exp, hidden, freq//2+1, 1, 2
        self.seq_pool = nn.AvgPool1d(args.max_seq_length)
        self.hid_pool2d = nn.AvgPool2d([1, args.hidden_size])
        self.hid_pool = nn.AvgPool1d(args.hidden_size)
        self.freq_pool = nn.AvgPool1d(freqs * 2)
        if args.pooling == 1:
            self.mlp = nn.Linear(args.hidden_size, self.expert_num)
        elif args.pooling == 2:
            self.mlp = nn.Linear(args.max_seq_length, self.expert_num)
        elif args.pooling == 3:
            self.mlp = nn.Linear(freqs * 2, self.expert_num)
        self.pooling = args.pooling
        self.filter_vary = args.filter_vary
        self.trunc_gate = args.trunc_gate

    def forward(self, input_tensor, attention_mask=None):
        batch, seq_len, hidden = input_tensor.shape
        x = input_tensor.transpose(1, 2)
        stft = torch.stft(
            x.contiguous().view(batch * hidden, seq_len),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            normalized=self.normalized,
            center=self.center,
            window=self.window,
            return_complex=True,
        )
        _, freqs, intervals = stft.shape
        stft = stft.reshape(batch, hidden, freqs, intervals)

        # gating network
        if self.pooling == 1:
            gate = self.seq_pool(x).squeeze()
        elif self.pooling == 2:
            gate = self.hid_pool(input_tensor).squeeze()  # [B, T]
        elif self.pooling == 3:
            real = stft.real.permute(0, 3, 2, 1)  # [B, interval, freq//2, hidden]
            imag = stft.imag.permute(0, 3, 2, 1)
            gate = torch.cat(
                [self.hid_pool2d(real).squeeze(), self.hid_pool2d(imag).squeeze()],
                dim=-1,
            )  # [B, interval, freq] eliminate H
        if self.trunc_gate == 1:  # truncate the gradient to embedding layer
            gate = torch.detach(gate)
        gate = self.mlp(gate)
        gate = torch.softmax(2 * gate, dim=-1)

        if self.pooling == 3:
            weight = (
                torch.einsum("imj, jklmn->iklmn", gate, self.filter) / self.expert_num
            )  # [time, gate], [gate, hidden, freq//2+1, time, 2]
        else:
            weight = (
                torch.einsum("ij, jklmn->iklmn", gate, self.filter) / self.expert_num
            )

        weight = torch.view_as_complex(weight)
        stft = stft * weight
        stft = stft.reshape(batch * hidden, freqs, intervals)
        istft = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            onesided=self.onesided,
            length=seq_len,
            normalized=self.normalized,
            center=self.center,
            window=self.window,
            return_complex=False,
        )
        istft = istft.reshape(batch, hidden, seq_len)
        istft = istft.transpose(1, 2).contiguous()
        hidden_states = self.out_dropout(istft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class ProbAttention(nn.Module):
    """
    Canonical SAR
    """

    def __init__(self, args):
        super(ProbAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (args.hidden_size, args.num_attention_heads)
            )

        self.mapping_L = nn.Linear(args.hidden_size, args.inner_size)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)
        self.use_softmax = args.use_softmax
        self.use_value_mapping = args.use_value_mapping
        
    def calculate_attention_matrix(self, L, k):
        """
        Calculate the attention matrix D based on Determinantal Point Processes (DPP).

        Args:
            L (torch.Tensor): A batch of DPP kernels with shape (batch_size, T, T).
            k (int): The size of subsets to consider.

        Returns:
            torch.Tensor: The attention matrix D with shape (batch_size, T, T).
        """
        import itertools
        batch_size, T, _ = L.shape

        subsets = list(itertools.combinations(range(T), k))  # List of tuples
        C = len(subsets)  # Number of combinations C(T, k)
        subsets = torch.tensor(subsets, dtype=torch.long, device=L.device)  # Shape: (C, k)

        det_LS = torch.zeros(batch_size, C, device=L.device)  # Shape: (batch_size, C)
        for idx, subset in enumerate(subsets):
            L_S = L[:, subset, :][:, :, subset]
            det = torch.det(L_S)  # Shape: (batch_size,)
            det_LS[:, idx] = det  # Store determinants

        sum_det = det_LS.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
        p_S = det_LS / sum_det  # Shape: (batch_size, C)
        p_S = torch.exp(-1*p_S)

        D = torch.zeros(batch_size, T, T, device=L.device)  # Shape: (batch_size, T, T)
        for idx, subset in enumerate(subsets):
            # Probability for the current subset across the batch
            p = p_S[:, idx].unsqueeze(1)  # Shape: (batch_size, 1)
            indices = subset.tolist()
            D[:, indices[0], indices[1]] += p[:, 0]
            D[:, indices[1], indices[0]] += p[:, 0]

        return D
    def calculate_attention_matrix_optimized(self, L, k):
        """
        Optimized calculation of the attention matrix D based on Determinantal Point Processes (DPP).
        
        Args:
            L (torch.Tensor): A batch of DPP kernels with shape (batch_size, T, T).
            k (int): The size of subsets to consider.
        
        Returns:
            torch.Tensor: The attention matrix D with shape (batch_size, T, T).
        """
        import itertools
        batch_size, T, _ = L.shape

        subsets = list(itertools.combinations(range(T), k))  # List of C subsets
        C = len(subsets)  # Number of combinations C(T, k)
        subsets_tensor = torch.tensor(subsets, dtype=torch.long, device=L.device)  # Shape: (C, k)

        # Step 2: Extract all relevant submatrices L_S in a batched manner
        # L_subsets: (batch_size, C, k, k)
        # First, gather the rows corresponding to each subset
        idx_i, idx_j = zip(*subsets_tensor)
        
        M_ii = L[:, idx_i, idx_i]  # Shape: (B, K)
        M_ij = L[:, idx_i, idx_j]
        M_ji = L[:, idx_j, idx_i]
        M_jj = L[:, idx_j, idx_j]
    
        L_subsets = torch.stack([
            torch.stack([M_ii, M_ij], axis=-1),
            torch.stack([M_ji, M_jj], axis=-1)
        ], axis=-2)
        # Step 3: Compute determinants of all submatrices
        # Reshape for batch determinant computation
        L_subsets_reshaped = L_subsets.view(batch_size * C, k, k)  # Shape: (batch_size * C, k, k)
        det_LS = torch.det(L_subsets_reshaped).view(batch_size, C)  # Shape: (batch_size, C)
        print("det", det_LS.mean())

        # Step 4: Normalize determinants to obtain probabilities p_S
        sum_det = det_LS.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
        p_S = torch.exp(-1*det_LS / (sum_det+1e-5))  # Shape: (batch_size, C)
        # Step 5: Prepare indices for scatter_add_
        # Generate all (i, j) pairs for each subset
        # subsets_tensor: (C, k)
        # Create grid of indices for each subset
        subsets_i = subsets_tensor.unsqueeze(2).expand(-1, -1, k).reshape(C * k * k)  # Shape: (C*k*k,)
        subsets_j = subsets_tensor.unsqueeze(1).expand(-1, k, -1).reshape(C * k * k)  # Shape: (C*k*k,)

        # Compute flattened indices for scatter_add_
        # Each (i, j) pair maps to a unique index in the flattened T*T matrix
        indices = subsets_i * T + subsets_j  # Shape: (C*k*k,)

        # Step 6: Repeat p_S for all (i, j) pairs
        # p_S: (batch_size, C)
        # Each p_S[b, c] needs to be added k*k times for its subset
        p_S_repeated = p_S.unsqueeze(2).repeat(1, 1, k * k).reshape(batch_size, C * k * k)  # Shape: (batch_size, C*k*k)

        # Step 7: Initialize the flattened attention matrix D
        D_flat = torch.zeros(batch_size, T * T, device=L.device)  # Shape: (batch_size, T*T)

        # Step 8: Perform scatter_add_ to accumulate probabilities into D_flat
        # The scatter_add_ operation adds p_S_repeated to the appropriate positions in D_flat
        D_flat.scatter_add_(1, indices.unsqueeze(0).expand(batch_size, -1), p_S_repeated)

        # Step 9: Reshape D_flat back to (batch_size, T, T)
        D = D_flat.view(batch_size, T, T)  # Shape: (batch_size, T, T)
        print("D", D.mean())
        return D
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        S = self.mapping_L(input_tensor)
        L = torch.matmul(S, S.transpose(-1, -2))

        attention_probs = self.calculate_attention_matrix_optimized(L, 2)
        print(attention_probs.mean())
        if self.use_softmax == 1:
            attention_probs = nn.Softmax(dim=-1)(attention_probs)
        else:
            attention_probs = attention_probs
        # attention_probs = self.calculate_attention_matrix(L, 2)
        attention_probs = self.attn_dropout(attention_probs)
        if self.use_value_mapping == 1:
            hidden_states = torch.matmul(attention_probs, self.value(input_tensor))
        else:
            hidden_states = torch.matmul(attention_probs, input_tensor)
        # hidden_states = self.dense(hidden_states)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states



class Intermediate(nn.Module):
    """
    The efficacy of FFN remains to be explored.
    """

    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(
            args.hidden_size, args.hidden_size * args.ffn_multiplier
        )
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(
            args.ffn_multiplier * args.hidden_size, args.hidden_size
        )
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.ablate = args.ablate

    def forward(self, input_tensor):
        if self.ablate == 6:
            return input_tensor
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class MetaFormerBlock(nn.Module):
    """
    Construct a metaFormer layer with args.model_name
    """

    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.intermediate = Intermediate(args)
        filter_models = {
            "FMLP": FilterLayer(args),
            "STFT": STFTLayer(args),
            "STFTW": STFTLayerW(args),
            "STFTWM": STFTLayerWM(args),
            "FMLPM1": FilterLayerM1(args),
            "FMLPM11": FilterLayerM11(args),
            "FMLPM12": FilterLayerM12(args),
            "FMLPM2": FilterLayerM2(args),
            "FMLPM3": FilterLayerM3(args),
            # 'WAVE': WaveletLayer(args),
            "CONV": ConvLayer(args),
            "FCONV": FConvLayer(args),
            "SpecCross": SpecCrossLayer(args),
            "Filter2D": FilterLayer2D(args),
            "Linear": LineLayer(args),
        }

        attention_models = {
            "SASRec": SelfAttention(args),
            "STFT": STFTLayer(args),
            "STFTW": STFTLayerW(args),
            "STFTWM": STFTLayerWM(args),
            "SyntV1": SynthesisV1(args),
            "SyntV11": SynthesisV11(args),
            "SyntV12": SynthesisV12(args),
            "SyntV2": SynthesisV2(args),
            "SyntV3": SynthesisV3(args),
            "SyntV4": SynthesisV4(args),
            "SyntV5": SynthesisV5(args),
            "LocalSelfAttn": LocalSelfAttention(args),
            "FastSelfAttn": FastSelfAttention(args),
            "PoolingSelfAttn": PoolingSelfAttention(args),
            "NoSharedSelfAttn": NoSharedSelfAttention(args),
            "OverfitSelfAttn": OverfitSelfAttention(args),
            "ProbSelfAttn": ProbAttention(args),
        }
        if self.model_name in filter_models:
            self.filter = filter_models[args.model_name]
            self.use_mask = False
        if self.model_name in attention_models:
            self.attention = attention_models[args.model_name]
            self.use_mask = True

    def forward(self, hidden_states, attention_mask):
        if self.use_mask is True:
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filter(hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output


class CananicalBlock(nn.Module):
    """
    Construct a layer with args.model_name, i.e., without FFN.
    """

    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        filter_models = {
            "FMLP": FilterLayer(args),
            "STFT": STFTLayer(args),
            "STFTW": STFTLayerW(args),
            "STFTWM": STFTLayerWM(args),
            "FMLPM1": FilterLayerM1(args),
            "FMLPM11": FilterLayerM11(args),
            "FMLPM12": FilterLayerM12(args),
            "FMLPM2": FilterLayerM2(args),
            "FMLPM3": FilterLayerM3(args),
            # 'WAVE': WaveletLayer(args),
            "CONV": ConvLayer(args),
            "FCONV": FConvLayer(args),
            # "FilterCross": FilterCrossLayer(args),
            # "FilterCross2": FilterCrossLayer2(args),
            "Filter2D": FilterLayer2D(args),
            "GRU": GRULayer(args),
            "CanonicalGRU": CanonicalGRULayer(args),
            "LMU": LMULayer(args),
        }

        attention_models = {
            "SASRec": SelfAttention(args),
            "STFT": STFTLayer(args),
            "STFTW": STFTLayerW(args),
            "STFTWM": STFTLayerWM(args),
            "SyntV1": SynthesisV1(args),
            "SyntV11": SynthesisV11(args),
            "SyntV12": SynthesisV12(args),
            "SyntV2": SynthesisV2(args),
            "SyntV3": SynthesisV3(args),
            "SyntV4": SynthesisV4(args),
            "SyntV5": SynthesisV5(args),
            "LocalSelfAttn": LocalSelfAttention(args),
            "FastSelfAttn": FastSelfAttention(args),
            "PoolingSelfAttn": PoolingSelfAttention(args),
            "NoSharedSelfAttn": NoSharedSelfAttention(args),
            "OverfitSelfAttn": OverfitSelfAttention(args),
        }
        if self.model_name in filter_models:
            self.filter = filter_models[args.model_name]
            self.use_mask = False
        if self.model_name in attention_models:
            self.attention = attention_models[args.model_name]
            self.use_mask = True

    def forward(self, hidden_states, attention_mask):
        if self.use_mask is True:
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filter(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = (
            MetaFormerBlock(args) if args.use_metaformer == 1 else CananicalBlock(args)
        )
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
