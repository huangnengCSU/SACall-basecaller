# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LossCompute(nn.Module):
    def __init__(self, blank=0):
        super(LossCompute, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank)

    def forward(self, log_probs, gold, input_lengths, target_lengths):
        gold = gold.contiguous().view(-1)
        loss = self.ctc_loss(log_probs, gold, input_lengths, target_lengths)
        return loss
