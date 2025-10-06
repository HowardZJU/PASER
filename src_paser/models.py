import torch
import torch.nn as nn
from modules import Encoder
from cnn import *
import math


class PASModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.zero_pad = args.zero_pad
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        # self.scale_agg = nn.Linear(args.hidden_size * args.num_hidden_layers, args.hidden_size)
        self.add_attnmixer()
        self.apply(self.init_weights)

    def add_attnmixer(self):
        if self.args.use_attn_mixer == 1:
            self.attn_mixer = AttnMixer(
                mix_set_size=self.args.mix_set_size,
                num_attention_heads=self.args.mix_num_attention_heads,
                hidden_size=self.args.hidden_size,
                dropout_rate=self.args.hidden_dropout_prob,
            )

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        if self.args.ablate == 1 or self.args.use_pos_emb == 0:
            sequence_emb = item_embeddings
        else:
            sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.layernorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).bool()  # non-masked
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = ~torch.triu(
            torch.ones(attn_shape, device=input_ids.device), diagonal=1
        ).bool()  # torch.uint8
        subsequent_mask = subsequent_mask.unsqueeze(1)

        extended_attention_mask = extended_attention_mask & subsequent_mask
        if self.args.model_name == "FastSelfAttn":
            extended_attention_mask = attention_mask.unsqueeze(1)  # torch.int64

        sequence_emb = self.add_position_embedding(input_ids)
        if self.zero_pad == 1:
            # sequence_emb = sequence_emb.masked_fill(~attention_mask.unsqueeze(2), value=torch.tensor(0.0))
            sequence_emb = sequence_emb.masked_fill(
                ~attention_mask.unsqueeze(2), value=1e-8
            )
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        # if self.args.multiscale == 0:
        sequence_output = item_encoded_layers[-1]
        # elif self.args.multiscale == 1:
        #     sequence_output = sum(item_encoded_layers) / self.args.num_hidden_layers
        # elif self.args.multiscale == 2:
        #     # sequence_output = torch.stack(item_encoded_layers).max(0)[0]
        #     sequence_output = torch.stack(item_encoded_layers, dim=2).flatten(2)
        #     sequence_output = self.scale_agg(sequence_output)
        if self.args.use_attn_mixer == 1:
            sequence_output = self.attn_mixer(sequence_output, attention_mask)

        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class AttnMixer(nn.Module):
    """
    AttnMixer
    References: https://arxiv.org/pdf/2206.12781.pdf
    """

    def __init__(self, mix_set_size, num_attention_heads, hidden_size, dropout_rate):
        super().__init__()
        self.set_size = mix_set_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # TODO compare the implementation
        self.query_generateor = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False)
                for i in range(self.set_size)
            ]
        )  # W_{ql}
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.lppool = torch.nn.LPPool1d(4, self.set_size, stride=self.set_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_aff = nn.Linear(hidden_size * 2, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        batch, seq_len, hidden = input_tensor.shape
        # Multi-level User Intent Generation.
        input_tensor = input_tensor.div(
            torch.norm(input_tensor, p=2, dim=2, keepdim=True) + 1e-8
        )  # [B, seq, hid]
        input_tensor_cum = torch.cumsum(torch.flip(input_tensor, [1]), dim=1)
        input_tensor_cum = input_tensor_cum[:, : self.set_size, :]
        query = [
            self.query_generateor[i](input_tensor_cum[:, i, :])
            for i in range(self.set_size)
        ]  # [ [B,H], [B,H] ... ]
        query = torch.stack(query, dim=1)  # [B, L, H]
        query = self.query(query)
        query_layer = self.transpose_for_scores(query)
        key = self.key(input_tensor)
        key_layer = self.transpose_for_scores(key)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = torch.softmax(
            2 * attention_scores, dim=-1
        )  # [B, Head, L, Seq]

        # Attention Mixture to Generate Session Embeddings, Eq.(8)
        attention_probs_reshape = attention_probs.reshape(
            -1, attention_probs.size()[2], seq_len
        ).transpose(
            1, 2
        )  # [B*Head, Seq, L]
        alpha_hat = (
            self.lppool(attention_probs_reshape).squeeze().reshape(batch, -1, seq_len)
        )  # [B, Head, Seq]
        alpha_hat = alpha_hat.masked_fill(
            ~attention_mask.unsqueeze(1), -1e-9
        )  # the existing tokens
        alpha = torch.softmax(2 * alpha_hat, dim=1)
        alpha = self.dropout(alpha)

        s = torch.einsum("bes,bsh->bh", alpha_hat, input_tensor)
        s = s.div(torch.norm(s, p=2, dim=1, keepdim=True) + 1e-8)
        out = self.final_aff(torch.cat([s, input_tensor[:, -1, :]], dim=-1))
        out = self.dropout(out)

        return out.unsqueeze(1)  # [B, 1, H]
