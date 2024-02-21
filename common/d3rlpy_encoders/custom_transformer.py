import dataclasses
import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory


class PointWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            # nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, inputs):
        # outputs = self.net(inputs.transpose(-1, -2))
        # outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs = self.net(inputs)
        # outputs += inputs
        return inputs + outputs


class TransformerEncoder(nn.Module):
    def __init__(self,
                 observation_shape,
                 embed_size,
                 hidden_size,
                 time_len,
                 num_blocks,
                 num_heads,
                 ffn_dropout,
                 embed_dropout,
        ):
        super().__init__()
        input_size = observation_shape[1]
        self.pre_layer = nn.Linear(input_size, embed_size)
        self.hidden_size = hidden_size

        ## core structures
        self.attn_layer_norms = nn.ModuleList() # to be Q for self-attention
        self.attn_layers = nn.ModuleList()
        self.forward_layer_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            attn_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)
            self.attn_layer_norms.append(attn_layer_norm)
            attn_layer =  nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
            self.attn_layers.append(attn_layer)
            fwd_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)
            self.forward_layer_norms.append(fwd_layer_norm)
            fwd_layer = PointWiseFeedForward(embed_size, hidden_size, ffn_dropout)
            self.forward_layers.append(fwd_layer)
        self.last_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)

        self.pos_embedding = nn.Embedding(time_len, embed_size)
        self.embed_dropout = nn.Dropout(embed_dropout)
    
    def forward(self, x):
        timeline_mask = torch.BoolTensor(x.sum(dim=-1).cpu() != 0).to(x.device)

        x = self.pre_layer(x)
        pos_ids = torch.arange(x.shape[1]).expand(x.shape[0], -1).to(x.device)
        pos_embed = self.embed_dropout(self.pos_embedding(pos_ids))
        x_with_pos = x + pos_embed
        x = x_with_pos * timeline_mask.unsqueeze(2) # broadcast in last dim

        time_len = x.shape[1]
        attention_mask = ~torch.tril(torch.ones((time_len, time_len), dtype=torch.bool, device=x.device))
        for i in range(len(self.attn_layers)):
            Q = self.attn_layer_norms[i](x)
            attn_outputs, _ = self.attn_layers[i](Q, x, x,
                                                # key_padding_mask=timeline_mask,  #[TODO] loss to be nan
                                                attn_mask=attention_mask)
            x = Q + attn_outputs

            x = self.forward_layer_norms[i](x)
            x = self.forward_layers[i](x)
            x *= timeline_mask.unsqueeze(2)

        return self.last_layer_norm(x)[:, -1, :]


class TransformerEncoderWithAction(nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 embed_size,
                 hidden_size,
                 time_len,
                 num_blocks,
                 num_heads,
                 ffn_dropout,
                 embed_dropout,
        ):
        super().__init__()
        input_size = observation_shape[1]
        self.pre_layer = nn.Linear(input_size, embed_size)
        self.hidden_size = hidden_size

        ## core structures
        self.attn_layer_norms = nn.ModuleList() # to be Q for self-attention
        self.attn_layers = nn.ModuleList()
        self.forward_layer_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            attn_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)
            self.attn_layer_norms.append(attn_layer_norm)
            attn_layer =  nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
            self.attn_layers.append(attn_layer)
            fwd_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)
            self.forward_layer_norms.append(fwd_layer_norm)
            fwd_layer = PointWiseFeedForward(embed_size, hidden_size, ffn_dropout)
            self.forward_layers.append(fwd_layer)
        self.last_layer_norm = nn.LayerNorm(embed_size, eps=1e-8)

        self.pos_embedding = nn.Embedding(time_len, embed_size)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # self.action_layer = nn.Sequential(
        #     PointWiseFeedForward(action_size, hidden_size, ffn_dropout),
        #     nn.LayerNorm(action_size, eps=1e-8)
        # )
        self.fc_action = nn.Linear(action_size, embed_size)
        self.fc_concat = nn.Linear(embed_size * 2, hidden_size)
    
    def forward(self, x, action):
        timeline_mask = torch.BoolTensor(x.sum(dim=-1).cpu() != 0).to(x.device)

        x = self.pre_layer(x)
        pos_ids = torch.arange(x.shape[1]).expand(x.shape[0], -1).to(x.device)
        pos_embed = self.embed_dropout(self.pos_embedding(pos_ids))
        x_with_pos = x + pos_embed
        x = x_with_pos * timeline_mask.unsqueeze(2) # broadcast in last dim

        time_len = x.shape[1]
        attention_mask = ~torch.tril(torch.ones((time_len, time_len), dtype=torch.bool, device=x.device))
        for i in range(len(self.attn_layers)):
            Q = self.attn_layer_norms[i](x)
            attn_outputs, _ = self.attn_layers[i](Q, x, x,
                                                # key_padding_mask=timeline_mask,  #[TODO] loss to be nan
                                                attn_mask=attention_mask)
            x = Q + attn_outputs

            x = self.forward_layer_norms[i](x)
            x = self.forward_layers[i](x)
            x *= timeline_mask.unsqueeze(2)

        encoded_x = self.last_layer_norm(x)[:, -1, :]
        concat = torch.cat([encoded_x, torch.relu(self.fc_action(action))], dim=1)
        return torch.relu(self.fc_concat(concat))


@dataclasses.dataclass()
class TransformerEncoderFactory(EncoderFactory):
    embed_size: int = 100
    hidden_size: int = 200
    time_len: int = 30
    num_blocks: int = 2
    num_heads: int = 4
    ffn_dropout: float = 0.1
    embed_dropout: float = 0.1

    def create(self, observation_shape):
        print('Observation_shape: ', observation_shape)
        return TransformerEncoder(observation_shape, self.embed_size, self.hidden_size, self.time_len, 
                            self.num_blocks, self.num_heads, self.ffn_dropout, self. embed_dropout)
    
    def create_with_action(self, observation_shape, action_size):
        # print(observation_shape, action_size)
        print('Action size: ', action_size)
        return TransformerEncoderWithAction(observation_shape, action_size, self.embed_size, self.hidden_size, self.time_len,
                                    self.num_blocks, self.num_heads, self.ffn_dropout, self. embed_dropout)

    @staticmethod
    def get_type() -> str:
        return "custom_transformer"
