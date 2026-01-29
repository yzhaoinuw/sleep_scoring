from einops import rearrange
from torch import nn

from .layers.head import cls_head
from .layers.transformer import SeqNewMoETransformer2, Transformer


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads
        seq_len = args.seq_len
        dropout = args.dropout
        path_drop = args.path_drop
        e_layers = args.e_layers
        ca_layers = args.ca_layers
        seq_layers = args.seq_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation
        n_sequences = args.n_sequences
        self.output_attentions = args.output_attentions
        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len
        mixffn_start_layer_index = seq_layers - ca_layers
        # self.stft_transform = STFT(win_length=patch_len,n_fft=256,hop_length=patch_len)
        self.eeg_transformer = Transformer(
            patch_len,
            n_patches,
            e_layers,
            c_in,
            inner_dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            path_drop=path_drop,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
            mix_type=args.mix_type,
            cls=True,
            flag="seq",
            domain="time",
            output_attentions=self.output_attentions,
        )

        self.emg_transformer = Transformer(
            patch_len,
            n_patches,
            e_layers,
            c_in,
            inner_dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            path_drop=path_drop,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
            mix_type=args.mix_type,
            cls=True,
            flag="seq",
            domain="time",
            output_attentions=self.output_attentions,
        )

        self.moe_transformer = SeqNewMoETransformer2(
            patch_len,
            n_sequences,
            seq_layers,
            c_in,
            inner_dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            path_drop=path_drop,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
            mix_type=args.mix_type,
            cls=False,
            flag="epoch",
            domain="time",
            mixffn_start_layer_index=mixffn_start_layer_index,
            output_attentions=False,
        )

        self.cls_head = cls_head(inner_dim, c_out)
        self.cls_head_eeg = cls_head(inner_dim, c_out)
        self.cls_head_emg = cls_head(inner_dim, c_out)

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = x[:, :, 0], x[:, :, 1]

        eeg, eeg_attn = self.eeg_transformer(eeg)
        emg, emg_attn = self.emg_transformer(emg)

        cls_eeg, cls_emg = eeg[:, :, -1], emg[:, :, -1]
        # x_our --> [b, n, 2d]

        infer = self.moe_transformer.infer(cls_eeg, cls_emg)
        logits = self.cls_head(infer["cls_feats"])
        infer_eeg = self.moe_transformer.infer_eeg(cls_eeg)
        infer_emg = self.moe_transformer.infer_emg(cls_emg)

        logits_eeg = self.cls_head_eeg(infer_eeg["cls_feats"])
        logits_emg = self.cls_head_emg(infer_emg["cls_feats"])

        logits = rearrange(logits, "b e d -> (b e) d")
        logits_eeg = rearrange(logits_eeg, "b e d -> (b e) d")
        logits_emg = rearrange(logits_emg, "b e d -> (b e) d")
        if label is not None:
            label = rearrange(label, "b e d -> (b e) d")

        out_dict = {
            "out": logits,
            "out_eeg": logits_eeg,
            "out_emg": logits_emg,
            "cls_feats": infer["cls_feats"],
            "cls_feats_eeg": infer_eeg["cls_feats"],
            "cls_feats_emg": infer_emg["cls_feats"],
            "raw_cls_feats": infer["raw_cls_feats"],
            "label": label,
        }
        return out_dict
