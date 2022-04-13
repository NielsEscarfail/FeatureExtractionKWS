# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO:::: CHANGE TO WAV2VEC 2


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


def generate_w2v_model_params(w2v_model_params):
    w2v_model_params.setdefault('extractor_mode', "default")
    w2v_model_params.setdefault('encoder_layers')
    w2v_model_params.setdefault('encoder_embed_dim', 768)
    w2v_model_params.setdefault('encoder_ffn_embed_dim', 3072)
    w2v_model_params.setdefault('encoder_attention_heads', 12)
    w2v_model_params.setdefault('activation_fn', "gelu")
    # dropouts
    w2v_model_params.setdefault('dropout', 0.1)
    w2v_model_params.setdefault('attention_dropout', 0.1)
    w2v_model_params.setdefault('activation_dropout', 0.0)
    w2v_model_params.setdefault('encoder_layerdrop', 0.0)
    w2v_model_params.setdefault('dropout_input', 0.0)
    w2v_model_params.setdefault('dropout_features', 0.0)
    w2v_model_params.setdefault('final_dim', 0)
    w2v_model_params.setdefault('layer_norm_first', False)
    w2v_model_params.setdefault('conv_feature_layers', "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]")
    w2v_model_params.setdefault('conv_bias', False)
    w2v_model_params.setdefault('logit_temp', 0.1)
    w2v_model_params.setdefault('quantize_targets', False)
    w2v_model_params.setdefault('quantize_input', False)
    w2v_model_params.setdefault('same_quantizer', False)
    w2v_model_params.setdefault('target_glu', False)
    w2v_model_params.setdefault('feature_grad_mult', 1.0)
    w2v_model_params.setdefault('latent_vars', 320)
    w2v_model_params.setdefault('latent_groups', 2)
    w2v_model_params.setdefault('latent_dim', 0)

    # masking
    w2v_model_params.setdefault('mask_length', 10)
    w2v_model_params.setdefault('mask_prob',0.65)
    w2v_model_params.setdefault('mask_selection', "static")
    w2v_model_params.setdefault('mask_other', 0)
    w2v_model_params.setdefault('no_mask_overlap', False)
    w2v_model_params.setdefault('mask_min_space', 1)

    # channel masking
    w2v_model_params.setdefault('mask_channel_length', 10)
    w2v_model_params.setdefault('mask_channel_prob', 0.0)
    w2v_model_params.setdefault('mask_channel_selection', "static")
    w2v_model_params.setdefault('mask_channel_other', 0)
    w2v_model_params.setdefault('no_mask_channel_overlap', False, )
    w2v_model_params.setdefault('mask_channel_min_space', 1)

    # negative selection
    w2v_model_params.setdefault('num_negatives', 100)
    w2v_model_params.setdefault('negatives_from_everywhere', False)
    w2v_model_params.setdefault('cross_sample_negatives', 0)
    w2v_model_params.setdefault('codebook_negatives', 0)

    # positional embeddings
    w2v_model_params.setdefault('conv_pos', 128)
    w2v_model_params.setdefault('conv_pos_groups', 16)

    w2v_model_params.setdefault('latent_temp', (2, 0.5, 0.999995))

    return w2v_model_params


"""@dataclass
class Wav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )"""


class Wav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg['conv_feature_layers'])
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg['extractor_mode'],
            conv_bias=cfg['conv_bias'],
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg['encoder_embed_dim'])
            if self.embed != cfg['encoder_embed_dim'] and not cfg['quantize_input']
            else None
        )

        self.mask_prob = cfg['mask_prob']
        self.mask_selection = cfg['mask_selection']
        self.mask_other = cfg['mask_other']
        self.mask_length = cfg['mask_length']
        self.no_mask_overlap = cfg['no_mask_overlap']
        self.mask_min_space = cfg['mask_min_space']

        self.mask_channel_prob = cfg['mask_channel_prob']
        self.mask_channel_selection = cfg['mask_channel_selection']
        self.mask_channel_other = cfg['mask_channel_other']
        self.mask_channel_length = cfg['mask_channel_length']
        self.no_mask_channel_overlap = cfg['no_mask_channel_overlap']
        self.mask_channel_min_space = cfg['mask_channel_min_space']

        self.dropout_input = nn.Dropout(cfg['dropout_input'])
        self.dropout_features = nn.Dropout(cfg['dropout_features'])

        self.feature_grad_mult = cfg['feature_grad_mult']

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg['num_negatives']
        self.cross_sample_negatives = cfg['cross_sample_negatives']
        self.codebook_negatives = cfg['codebook_negatives']
        self.negatives_from_everywhere = cfg['negatives_from_everywhere']

        self.logit_temp = cfg['logit_temp']

        final_dim = cfg['final_dim'] if cfg['final_dim'] > 0 else cfg['encoder_embed_dim']

        if cfg['quantize_targets']:
            vq_dim = cfg['latent_dim'] if cfg['latent_dim'] > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg['latent_vars'],
                temp=(2, 0.5, 0.999995),  # cfg['latent_temp'],
                groups=cfg['latent_groups'],
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg['quantize_input']:
            if cfg['same_quantizer'] and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg['latent_dim'] if cfg['latent_dim'] > 0 else cfg['encoder_embed_dim']
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg['latent_vars'],
                    temp=(2, 0.5, 0.999995),  # cfg['latent_temp'],
                    groups=cfg['latent_groups'],
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, cfg['encoder_embed_dim'])

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg['encoder_embed_dim']).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg['target_glu']:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg['encoder_embed_dim'], final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def forward(self, source, padding_mask=None, mask=True, features_only=False):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args['dropout']
        self.embedding_dim = args['encoder_embed_dim']

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args['conv_pos'],
            padding=args['conv_pos'] // 2,
            groups=args['conv_pos_groups'],
    )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args['conv_pos'] * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args['conv_pos']), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args['encoder_ffn_embed_dim'],
                    num_attention_heads=args['encoder_attention_heads'],
                    dropout=self.dropout,
                    attention_dropout=args['attention_dropout'],
                    activation_dropout=args['activation_dropout'],
                    activation_fn=args['activation_fn'],
                    layer_norm_first=args['layer_norm_first'],
                )
                for _ in range(args['encoder_layers'])
            ]
        )

        self.layer_norm_first = args['layer_norm_first']
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args['encoder_layerdrop']

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


"""
def generate_w2v_model_params(w2v_model_params):
    w2v_model_params.setdefault('offset', "auto")
    w2v_model_params.setdefault('prediction_steps', 12)
    w2v_model_params.setdefault('activation', "relu")

    w2v_model_params.setdefault('sample_distance',
                                None)  # "sample distance from target. does not work properly with cross-sampling"
    w2v_model_params.setdefault('cross_sample_negatives', 0)  # num of cross sampled negatives
    w2v_model_params.setdefault('num_negatives', 10)  # "num of cross sampled negatives"}
    w2v_model_params.setdefault('conv_feature_layers',
                                "[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]")
    # convolutional feature extraction layers [(dim, kernel_size, stride), ...]"
    w2v_model_params.setdefault('conv_aggregator_layers',
                                "[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), ""(512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), ""(512, 12, 1), (512, 13, 1)]")
    # convolutional aggregator layers [(dim, kernel_size, stride), ...]"
    w2v_model_params.setdefault('dropout', 0.0)
    w2v_model_params.setdefault('dropout_features', 0.0)
    w2v_model_params.setdefault('dropout_agg', 0.0)
    w2v_model_params.setdefault('aggregator', "cnn")
    w2v_model_params.setdefault('gru_dim', 512)
    w2v_model_params.setdefault('no_conv_bias', False)
    w2v_model_params.setdefault('agg_zero_pad', False)
    w2v_model_params.setdefault('skip_connections_feat', False)
    w2v_model_params.setdefault('skip_connections_agg', True)
    w2v_model_params.setdefault('residual_scale', 0.5)
    w2v_model_params.setdefault('log_compression', True)
    w2v_model_params.setdefault('balanced_classes', False)
    w2v_model_params.setdefault('project_features', "none")
    w2v_model_params.setdefault('non_affine_group_norm', False)
    w2v_model_params.setdefault('vq_type', "none")
    w2v_model_params.setdefault('vq_vars', 320)
    w2v_model_params.setdefault('vq_groups', 2)
    w2v_model_params.setdefault('vq_dim', 0)
    w2v_model_params.setdefault('vq_depth', 1)
    w2v_model_params.setdefault('combine_grups', False)
    w2v_model_params.setdefault('vq_temp', (2.0, 0.5, 0.999995))
    w2v_model_params.setdefault('vq_gamma', 0.25)  # gamma parameter for kmeans style vector quantization"
    w2v_model_params.setdefault('infonce', II("criterion.infonce"))
    return w2v_model_params


class Wav2VecModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        print("OCFJDSN ", cfg)

        self.prediction_steps = 12  # Number of steps to predict
        offset = "auto"  # if set to 'auto', it is computed automatically from the receptive field, else set to int value

        if cfg['activation'] == "relu":
            activation = nn.ReLU()
        elif cfg['activation'] == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + cfg['activation'])

        feature_enc_layers = eval(cfg['conv_feature_layers'])
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            log_compression=cfg['log_compression'],
            skip_connections=cfg['skip_connections_feat'],
            residual_scale=cfg['residual_scale'],
            non_affine_group_norm=cfg['non_affine_group_norm'],
            activation=activation,
        )
        embed = feature_enc_layers[-1][0]

        self.vector_quantizer = None
        if cfg['vq_type'] == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=embed,
                num_vars=cfg['vq_vars'],
                temp=cfg['vq_temp'],
                groups=cfg['vq_groups'],
                combine_groups=cfg['combine_groups'],
                vq_dim=cfg['vq_dim'] if cfg['vq_dim'] > 0 else embed,
                time_first=False,
                activation=activation,
                weight_proj_depth=cfg['vq_depth'],
                weight_proj_factor=2,
            )
        elif cfg['vq_type'] == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=embed,
                num_vars=cfg.vq_vars,
                groups=cfg.vq_groups,
                combine_groups=cfg.combine_groups,
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
                time_first=False,
                gamma=cfg.vq_gamma,
            )
        else:
            assert (
                    cfg['vq_type'] == "none" or cfg['vq_type'] is None
            ), "Unknown quantizer type"

        if cfg['offset'] == "auto":
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)

        offset = int(offset)

        def make_aggregator():
            if cfg['aggregator'] == "cnn":
                agg_layers = eval(cfg['conv_aggregator_layers'])
                agg_dim = agg_layers[-1][0]
                feature_aggregator = ConvAggegator(
                    conv_layers=agg_layers,
                    embed=embed,
                    dropout=cfg['dropout'],
                    skip_connections=cfg['skip_connections_agg'],
                    residual_scale=cfg['residual_scale'],
                    non_affine_group_norm=cfg['non_affine_group_norm'],
                    conv_bias=not cfg['no_conv_bias'],
                    zero_pad=cfg['agg_zero_pad'],
                    activation=activation,
                )
            elif cfg['aggregator'] == "gru":
                agg_dim = cfg['gru_dim']
                feature_aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=embed,
                        hidden_size=agg_dim,
                        num_layers=1,
                        dropout=cfg['dropout'],
                    ),
                    TransposeLast(deconstruct_idx=0),
                )
            else:
                raise Exception("unknown aggregator type " + cfg['aggregator'])

            return feature_aggregator, agg_dim

        self.feature_aggregator, agg_dim = make_aggregator()

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=cfg['prediction_steps'],
            n_negatives=cfg['num_negatives'],
            cross_sample_negatives=cfg['cross_sample_negatives'],
            sample_distance=cfg['sample_distance'],
            dropout=cfg['dropout'],
            offset=offset,
            balanced_classes=cfg['balanced_classes'],
            infonce=cfg['infonce'],
        )

        self.dropout_feats = nn.Dropout(p=cfg['dropout_features'])
        self.dropout_agg = nn.Dropout(p=cfg['dropout_agg'])

        if cfg['project_features'] == "none":
            self.project_features = None
        elif cfg['project_features'] == "same":
            self.project_features = self.feature_aggregator
        elif cfg['project_features'] == "new":
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)

        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """"""Maximum length supported by the model.""""""
        return sys.maxsize

    def get_logits(self, net_output):
        logits = net_output["cpc_logits"]
        return logits

    def get_targets(self, sample, net_output):
        t = net_output["cpc_targets"]
        if isinstance(t, tuple):
            t = t[0]
        return t.contiguous()

    def get_target_weights(self, targets, net_output):
        targets = net_output["cpc_targets"]
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]
        return None

    def get_extra_losses(self, net_output):
        loss = None
        if "prob_perplexity" in net_output:
            loss = net_output["num_vars"] - net_output["prob_perplexity"]
        elif "kmeans_loss" in net_output:
            loss = net_output["kmeans_loss"]

        return loss


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers,
            dropout,
            log_compression,
            skip_connections,
            residual_scale,
            non_affine_group_norm,
            activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class ConvAggegator(nn.Module):
    def __init__(
            self,
            conv_layers,
            embed,
            dropout,
            skip_connections,
            residual_scale,
            non_affine_group_norm,
            conv_bias,
            zero_pad,
            activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class Wav2VecPredictionsModel(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            prediction_steps,
            n_negatives,
            cross_sample_negatives,
            sample_distance,
            dropout,
            offset,
            balanced_classes,
            infonce,
    ):
        super().__init__()

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):

        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)

        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)

        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )
        if self.infonce:
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            labels = torch.zeros_like(predictions)
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                labels[start: start + pos_num] = 1.0
                if weights is not None:
                    weights[start: start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        if self.infonce:
            predictions = predictions.view(-1, copies)
        else:
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels
"""
