"""论文候选模型的可训练重建架构，具体层数与原实现未能核实。"""

from typing import Dict, List
import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel-1)*dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self.padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding:
            y = y[:,:,:-self.padding]
        return x + self.dropout(torch.relu(y))


class BiTCN(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: List[int], dropout: float):
        super().__init__()
        self.forward_net = nn.Sequential(*[CausalBlock(channels,kernel,d,dropout) for d in dilations])
        self.backward_net = nn.Sequential(*[CausalBlock(channels,kernel,d,dropout) for d in dilations])
        self.merge = nn.Conv1d(2*channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_net(x)
        bwd = self.backward_net(x.flip(-1)).flip(-1)
        return self.merge(torch.cat([fwd,bwd],dim=1))


class SignalNetwork(nn.Module):
    def __init__(self, architecture: str, input_channels: int, outputs: int, config: Dict):
        super().__init__()
        h,k,d = config["hidden_dim"],config["kernel_size"],config["dropout"]
        self.architecture = architecture
        self.project = nn.Conv1d(input_channels,h,1)
        self.cnn = nn.Sequential(nn.Conv1d(h,h,k,padding=k//2),nn.ReLU(),nn.Conv1d(h,h,k,padding=k//2),nn.ReLU(),nn.Dropout(d))
        self.bitcn = BiTCN(h,k,config["tcn_dilations"],d)
        self.tcn = nn.Sequential(*[CausalBlock(h,k,v,d) for v in config["tcn_dilations"]])
        if architecture == "RNN":
            self.recurrent = nn.RNN(h,h,batch_first=True)
        else:
            self.recurrent = nn.LSTM(h,h,batch_first=True,bidirectional=architecture in ("TCN-Bi-LSTM","CNN-Bi-LSTM"))
        width = 2*h if architecture in ("TCN-Bi-LSTM","CNN-Bi-LSTM") else h
        self.attention = nn.Linear(width,1)
        self.head = nn.Linear(width,outputs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        y = self.project(x.transpose(1,2))
        if self.architecture in ("CNN","1D-CNN","CBTA","CNN-Bi-LSTM"):
            y = self.cnn(y)
        if self.architecture in ("BiTCN","CBTA"):
            y = self.bitcn(y)
        if self.architecture == "TCN-Bi-LSTM":
            y = self.tcn(y)
        y = y.transpose(1,2)
        if self.architecture in ("LSTM","AttentionLSTM","RNN","TCN-Bi-LSTM","CNN-Bi-LSTM"):
            y,_ = self.recurrent(y)
        if self.architecture in ("CBTA","AttentionLSTM"):
            weights = torch.softmax(self.attention(y),dim=1)
            return (y*weights).sum(1)
        return y.mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
