import math
import typing as ty

import torch
import torch.nn.init as nn_init
from torch import Tensor, nn
from torch.nn import functional as F


class EMA:
    def __init__(self, parameter, decay=0.99):
        self.parameter = parameter
        self.decay = decay

    def update(self, current_data):
        with torch.no_grad():
            self.parameter = self.parameter.mul_(self.decay).add_((1 - self.decay) * current_data)

class Linear_relu(nn.Linear):
    def __init__(self, in_features, out_features, act_type, bias=True, device=None, dtype=None):
        super(Linear_relu, self).__init__(
            in_features, out_features, bias, device, dtype
        )
        if act_type == "relu":
            self.act_fn = nn.ReLU()
        elif act_type == "prelu":
            self.act_fn = nn.PReLU()

    def forward(self, x):
        return self.act_fn(super().forward(x))

class LinearEmbeddings(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        hidden_size: int,
        add_act: bool,
        bias: bool,
        act_type: str,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), hidden_size)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, hidden_size))
        self.bias = nn.Parameter(Tensor(d_bias, hidden_size)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        if add_act:
            if act_type == "relu":
                self.act_fn = nn.ReLU()
            elif act_type == "prelu":
                self.act_fn = nn.PReLU()
        self.add_act = add_act

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        if self.bias:
            nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], include_cls_token: bool = True) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        if include_cls_token:
            x_num = torch.cat(
                [torch.ones(len(x_some), 1, device=x_some.device, dtype=x_num.dtype)]  # [CLS]
                + ([] if x_num is None else [x_num]),
                dim=1,
            )
            x = self.weight[None] * x_num[:, :, None]
        else:
            x_num = torch.cat([] + [x_num])
            x = self.weight[None, 1:, :] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            if include_cls_token:
                bias = torch.cat(
                    [
                        torch.zeros(1, self.bias.shape[1], device=x.device, dtype=x.dtype),
                        self.bias,
                    ]
                )
            else:
                bias = self.bias
            x = x + bias[None]
        if self.add_act:
            x = self.act_fn(x)

        return x

class EncoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout_ratio,
        act_type,
        add_act=False,
        if_bias=False,
        layer_idx=None,
        branch_idx=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.attention_dropout = dropout_ratio
        self.add_act = add_act
        linear_func = Linear_relu

        linear_hidden_size = self.hidden_size
        self.q_proj = linear_func(linear_hidden_size, linear_hidden_size, bias=if_bias, act_type=act_type)
        self.k_proj = linear_func(linear_hidden_size, linear_hidden_size, bias=if_bias, act_type=act_type)
        self.v_proj = linear_func(linear_hidden_size, linear_hidden_size, bias=if_bias, act_type=act_type)
        self.o_proj = linear_func(self.hidden_size, self.hidden_size, bias=if_bias, act_type=act_type)

        self.init_crross = 0
        self.linear_func = linear_func
        self.cross_proj = None
        if add_act:
            if act_type == "relu":
                self.act_fn = nn.ReLU()
            elif act_type == "prelu":
                self.act_fn = nn.PReLU()

        self.layer_idx = layer_idx
        self.branch_idx = branch_idx

    def forward(self, x, x_k=None, x_v=None):
        if x_k is None:
            x_k = x
        if x_v is None:
            x_v = x
        
        query_states = self.q_proj(x)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
        bsz, q_len, _ = x.size()
        k_bsz, k_len, _ = x_k.size()
        v_bsz, v_len, _ = x_v.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(k_bsz, k_len, self.num_heads, self.head_dim)
        value_states = value_states.view(
            v_bsz, v_len, self.num_heads, self.head_dim
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        
        if self.add_act:
            attn_output = self.act_fn(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class EncoderFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        dropout_ratio,
        act_type,
        if_bias=False,
        legacy=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout_ratio = dropout_ratio
        linear_func = Linear_relu
        if legacy:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=if_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=if_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=if_bias)
        else:
            self.gate_proj = linear_func(
                self.hidden_size, self.intermediate_size, bias=if_bias, act_type=act_type
            )
            self.up_proj = linear_func(
                self.hidden_size, self.intermediate_size, bias=if_bias, act_type=act_type
            )
            self.down_proj = linear_func(
                self.intermediate_size, self.hidden_size, bias=if_bias, act_type=act_type
            )
            
        self.drop_out = nn.Dropout(self.dropout_ratio)

        if act_type == "relu":
            self.act_fn = nn.ReLU()
        elif act_type == "prelu":
            self.act_fn = nn.PReLU()

    def forward(self, x):
        input_x = x
        if isinstance(self.gate_proj, Linear_relu):
            down_proj = (
                self.down_proj(self.gate_proj(input_x) + self.up_proj(input_x))
            )
        else:
            # using common implementation of transformer
            down_proj = self.down_proj(self.act_fn(self.gate_proj(input_x)) * self.up_proj(input_x)) 
        if self.training:
            down_proj = self.drop_out(down_proj)

        return down_proj

class MayaEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout_ratio,
        act_type,
        add_act=False,
        if_bias=False,
        layer_idx=0,
        skip_first_norm=False,
        if_last_layer=True,
        using_attn_norm=False,
        num_branch=3,
        label_nums=1,
        mlp_using_legacy=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        attn_hidden_size = hidden_size
        self.layer_idx = layer_idx
        real_num_heads = num_heads

        self.attn_layer_list = nn.ModuleList([])
        for i in range(num_branch):
            self.attn_layer_list.append(
                EncoderAttention(
                    attn_hidden_size,
                    real_num_heads,
                    dropout_ratio,
                    add_act=add_act,
                    if_bias=if_bias,
                    act_type=act_type,
                    layer_idx=self.layer_idx,
                    branch_idx=i
                )
            )
        ffn_hidden_size = hidden_size
        
        self.ffn_layer = EncoderFFN(
            ffn_hidden_size,
            intermediate_size,
            dropout_ratio,
            if_bias=if_bias,
            act_type=act_type,
            legacy=mlp_using_legacy
        )
        
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm_list = nn.ModuleList([])
        for i in range(num_branch):
            self.ffn_norm_list.append(nn.LayerNorm(hidden_size))

        self.init_crross = 0
        self.skip_first_norm = skip_first_norm
        self.if_last_layer = if_last_layer
        
        self.para_block_weights = nn.Parameter(torch.ones(num_branch))
        self.para_block_weights.requires_grad = False
        self.para_block_ema = EMA(self.para_block_weights, 0.8)

        self.using_attn_norm = using_attn_norm
        self.num_branch = num_branch
        self.gen_label = nn.Linear(hidden_size, label_nums, bias=if_bias)

        if label_nums > 1:
            self.loss_criterion = nn.CrossEntropyLoss()
        else:
            self.loss_criterion = nn.MSELoss()

    def forward(self, x, labels=None, candicate_x=None, candicate_y=None):
        attn_residual = x
       
        attn_inputs = []
        for i in range(self.num_branch):
            attn_inputs.append(x)
            
        res_x = []
        tmp_out_list = []
        for i in range(self.num_branch):
            tmp_x = attn_inputs[i]
            tmp_residual = tmp_x
            tmp_x = self.attn_layer_list[i](tmp_x, candicate_x, candicate_y)
            tmp_x = tmp_x + tmp_residual
            tmp_residual = tmp_x
            tmp_x = self.ffn_layer(tmp_x)
            tmp_x = self.ffn_norm_list[i](tmp_x)
            if self.training:
                if labels is not None:
                    if candicate_x is None:
                        tmp_out = self.gen_label(tmp_x[:, 0, :])
                    else:
                        tmp_out = self.gen_label(tmp_x.squeeze(0))
                    tmp_out = tmp_out.squeeze(1)
                    tmp_out = self.loss_criterion(tmp_out, labels)
                    tmp_out_list.append(tmp_out)
            res_x.append(tmp_x)
        if self.training:
            if tmp_out_list != []:
                tmp_out_tensor = torch.stack(tmp_out_list)
                current_para_block_weights = torch.nn.functional.softmax(tmp_out_tensor, dim=0) * self.num_branch
                self.para_block_ema.update(current_para_block_weights)
        res_x = torch.stack(res_x)
        weighted_res_x = res_x.clone()
        for i in range(self.num_branch):
            weighted_res_x[i] = res_x[i] * self.para_block_weights[i]
        x = torch.mean(weighted_res_x, dim=0)

        output_tensor = x[:, 0:1, :]
        x = (x + attn_residual) / 2
        if self.layer_idx == 0:
            x[:, 0:1, :] = output_tensor

        if self.using_attn_norm:
            x = self.attn_norm(x)

        return x

class DecoderL2Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        if_bias,
        qk_using_same_linear=True,
        using_scaling_L2=False
    ):
        super().__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=if_bias)
        if not qk_using_same_linear:
            self.Wk = nn.Linear(hidden_size, hidden_size, bias=if_bias)
        else:
            self.Wk = self.Wq
        self.Wv = nn.Linear(hidden_size, hidden_size, bias=if_bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=if_bias)
        self.dropout = dropout
        self.using_scaling_L2 = using_scaling_L2
    
    def forward(self, x_q, x_k, x_v, is_train):
        q, k, v = self.Wq(x_q), self.Wk(x_k), self.Wv(x_v)
        if self.using_scaling_L2:
            attention_weights = torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2) / torch.tensor(math.sqrt(q.shape[-1])).to(x_q.dtype)
        else:
            attention_weights = torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2).to(x_q.dtype).clone()
        if is_train and (self.Wk.weight == self.Wq.weight).all().item():
            attention_weights = attention_weights.squeeze()
            attention_weights = attention_weights.clone().fill_diagonal_(torch.inf)
            attention_weights = attention_weights[None, ...]
        attention_weights = F.softmax(-attention_weights, dim=-1)
        attention_weights = F.dropout(attention_weights, self.dropout, self.training)
        x = attention_weights @ v
        x = self.Wo(x)

        return x

class DecoderFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        dropout,
        if_bias,
        act_type
    ):
        super().__init__()
        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=if_bias)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=if_bias)
        self.W_down = nn.Linear(intermediate_size, hidden_size, bias=if_bias)
        self.dropout = dropout
        if act_type == "relu":
            self.act_fn = nn.ReLU()
        elif act_type == "prelu":
            self.act_fn = nn.PReLU()
    
    def forward(self, x):
        x_gate = self.act_fn(self.W_gate(x))
        x_gate = F.dropout(x_gate, self.dropout, self.training)
        x = self.W_down(x_gate * self.W_up(x))

        return x

class MayaDecoderLayer(nn.Module):
    def __init__(
        self,
        label_nums,
        decoder_hidden_size,
        decoder_intermediate_size,
        decoder_dropout_ratio,
        decoder_act_type,
        decoder_if_bias=False,
        qk_using_same_linear=True,
        using_scaling_L2=True,
        using_labels=True,
    ):
        super().__init__()
        self.attn = DecoderL2Attention(
            decoder_hidden_size,
            decoder_dropout_ratio,
            decoder_if_bias,
            qk_using_same_linear,
            using_scaling_L2
        )
        self.ffn = DecoderFFN(
            decoder_hidden_size,
            decoder_intermediate_size,
            decoder_dropout_ratio,
            decoder_if_bias,
            decoder_act_type
        )

        self.attn_norm = nn.LayerNorm(decoder_hidden_size)
        self.ffn_norm = nn.LayerNorm(decoder_hidden_size)
        self.label_encoder = (
            nn.Linear(1, decoder_hidden_size)
            if label_nums == 1
            else nn.Embedding(label_nums, decoder_hidden_size)
        )
        self.using_labels = using_labels  # for ablation study of IAIL without labels
    
    def forward(self, x, x_ref, labels, is_train):
        labels_emb = self.label_encoder(labels[..., None]).squeeze(-2)
        labels_emb = labels_emb[None, ...]
        x = x[None, ...]
        x_ref = x_ref[None, ...]
        
        x_attn_residual = x
        if self.using_labels:
            x = self.attn(x, x_ref, labels_emb, is_train)
        else:
            x = self.attn(x, x_ref, x_ref, is_train)
        x = x_attn_residual + x
        x = self.attn_norm(x)

        x_ffn_residual = x
        x = self.ffn(x)
        x = x_ffn_residual + x
        x = self.ffn_norm(x)

        return x

class MayaModel(nn.Module):
    def __init__(
        self,
        d_numerical,
        categories,
        hidden_size,
        num_heads,
        intermediate_size,
        num_layers,
        label_nums,
        dropout_ratio,
        act_type,
        add_act=False,
        if_bias=False,
        skip_first_norm=False,
        task_type="classification",
        last_mlp_skip=True,
        using_attn_norm=False,
        num_branch=3,
        mlp_using_legacy=False,
        using_encoder_decoder_arch=False,
        decoder_configs=None
    ):
        super().__init__()
        self.task_type = task_type

        if label_nums == 1:
            self.score_func = nn.Sigmoid()
        else:
            self.score_func = nn.Softmax(dim=1)
        
        self.emb = LinearEmbeddings(
            d_numerical=d_numerical, 
            categories=categories,
            hidden_size=hidden_size, 
            bias=True, 
            add_act=add_act, 
            act_type=act_type
        )
        self.gen_label = nn.Linear(hidden_size, label_nums, bias=if_bias)
        self.norm = nn.LayerNorm(hidden_size)
        
        encoder_list = []
        if_last_layer = False
        for layer_idx in range(num_layers):
            if last_mlp_skip and layer_idx == num_layers - 1:
                if_last_layer = True
            encoder_list.append(
                MayaEncoderLayer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    dropout_ratio,
                    add_act=add_act,
                    if_bias=if_bias,
                    layer_idx=layer_idx,
                    skip_first_norm=skip_first_norm,
                    if_last_layer=if_last_layer,
                    act_type=act_type,
                    using_attn_norm=using_attn_norm,
                    num_branch=num_branch,
                    label_nums=label_nums,
                    mlp_using_legacy=mlp_using_legacy
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_list)
        
        self.enable_decoder_arch = False
        if using_encoder_decoder_arch:
            num_decoder_layers = decoder_configs["num_decoder_layers"]
            decoder_configs["decoder_layer_configs"]["decoder_hidden_size"] = hidden_size
            self.decoder = nn.ModuleList(
                [
                    MayaDecoderLayer(
                        label_nums, **decoder_configs["decoder_layer_configs"]
                    ) for i in range(num_decoder_layers)
                ]
            )
            self.enable_decoder_arch=True
        else:
            self.decoder = None

    def encoder(self, x, labels=None):
        encoder_layer_res = []
        for idx, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, labels=labels)
            encoder_layer_res.append(x)
        x = torch.mean(torch.stack(encoder_layer_res), dim=0)
        x = self.norm(x)

        return x, encoder_layer_res

    def forward(
        self,  
        x_num: Tensor, 
        x_cat: ty.Optional[Tensor] = None, 
        labels=None, 
    ):
        x = self.emb(x_num, x_cat)
        x, encoder_layer_res = self.encoder(x, labels)
        sub_output_list = []
        
        cls_vec = x[:, 0, :]
        output = self.gen_label(cls_vec)
        for i in range(len(encoder_layer_res)):
            sub_out = encoder_layer_res[i][:, 0, :]
            sub_out = self.gen_label(sub_out)
            sub_out = sub_out.squeeze(-1)
            sub_output_list.append(sub_out)

        if self.enable_decoder_arch:
            if self.training:
                x = cls_vec
                x_ref = cls_vec
                for decoder_layer in self.decoder:
                    decoded_out = decoder_layer(
                        x, x_ref, labels, self.training
                    )
                    x = decoded_out
                output = self.gen_label(decoded_out)
                output = output.squeeze()

        output = output.squeeze(-1)
        if self.training == False and self.task_type == "classification":
            out_tensors = self.score_func(output)
        else:
            out_tensors = output

        if self.enable_decoder_arch and not self.training:
            return cls_vec
        else:
            return out_tensors
