import torch
import torch.nn as nn
import pytorch_lightning as pl


class MakeUp(pl.LightningModule):
    def __init__(self, params):
        super(MakeUp, self).__init__()
        self.type = params.pop('type')
        self.params = params
        if self.type == 'GRU':
            self.model = GRUAmp(**self.params)
        if self.type == 'Static':
            self.model = StaticAmp()

    def forward(self, x):
        return self.model(x)

    def reset_state(self, batch_size):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()


class GRUAmp(pl.LightningModule):
    def __init__(self, hidden_size):
        super(GRUAmp, self).__init__()
        self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)
        self.state = None

    def forward(self, x):
        res = x
        x, self.state = self.rec(x, self.state)
        x = self.lin(x)
        return x + res

    def reset_state(self, batch_size=None):
        self.state = None

    def detach_state(self):
        self.state = self.state.detach()


class StaticAmp(pl.LightningModule):
    def __init__(self):
        super(StaticAmp, self).__init__()
        self.cond = False
        self.gain = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, x, cond1=None, cond2=None):
        x = x + torch.tanh(self.bias)
        x = x*(torch.tanh(self.gain) + 1)
        return x

    def reset_state(self, batch_size=None):
        pass

    def detach_state(self):
        pass
