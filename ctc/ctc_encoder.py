import sys
sys.path.append('..')
import torch.nn as nn
import torch
from transformer.modules import PositionalEncoding, EncoderLayer
import generate_dataset.constants as Constants


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, num_encoder_layers, dropout=0.1):
        super(Encoder, self).__init__()
        # self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
        #                                          out_channels=d_model,
        #                                          kernel_size=1,
        #                                          stride=1,
        #                                          padding=0,
        #                                          bias=False),
        #                                nn.BatchNorm1d(num_features=d_model),
        #                                nn.ReLU(inplace=True))

        self.padding = 1
        self.kernel_size = 3
        self.stride = 2
        self.dilation = 1

        # self.pool_padding = 0
        # self.pool_kernel_size = 3
        # self.pool_stride = 2
        # self.pool_dilation = 1

        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
                                                 out_channels=d_model//2,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model//2),
                                       nn.ReLU(inplace=True),

                                       nn.Conv1d(in_channels=d_model//2,
                                                 out_channels=d_model,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       )
        # TODO: why padding_idx=0
        self.position_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout)

        self.stack_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_ff=d_ff, n_head=n_head, dropout=dropout) for _ in range(
                num_encoder_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, signal, signal_lengths):
        """
        :param signal: a tensor shape of [batch, length, 1]
        :param signal_lengths: a tensor shape of [batch,]
        :return:
        """
        # src_mask = signal.squeeze(2).eq(Constants.SIG_PAD).unsqueeze(-2)
        # print('old:',src_mask.size())
        max_len = signal.size(1)

        max_len = int(((max_len + 2 * self.padding - self.dilation * (
            self.kernel_size - 1) - 1) / self.stride + 1))

        max_len = int(((max_len + 2 * self.padding - self.dilation * (
            self.kernel_size - 1) - 1) / self.stride + 1))

        new_signal_lengths = ((signal_lengths + 2 * self.padding - self.dilation * (
            self.kernel_size - 1) - 1) / self.stride + 1).int()

        new_signal_lengths = ((new_signal_lengths + 2 * self.padding - self.dilation * (
            self.kernel_size - 1) - 1) / self.stride + 1).int()

        src_mask = torch.tensor([[0] * v.item() + [1] * (max_len - v.item()) for v in new_signal_lengths],
                                dtype=torch.uint8).unsqueeze(-2).to(signal.device)
        # print(src_mask)
        # print('new:',src_mask.size())
        signal = signal.transpose(-1, -2)  # (N,C,L)
        # print(signal.size())
        embed_out = self.src_embed(signal)  # (N,C,L)
        # print(embed_out.size())
        # print(embed_out[0])
        embed_out = embed_out.transpose(-1, -2)  # (N,L,C)
        enc_output = self.position_encoding(embed_out)
        # print(enc_output[0])
        # print(enc_output.size())
        # print(src_mask.size())
        for layer in self.stack_layers:
            enc_output, enc_slf_attn = layer(enc_output, src_mask)
            # print(enc_output[0])
        enc_output = self.layer_norm(enc_output)
        return enc_output, new_signal_lengths
