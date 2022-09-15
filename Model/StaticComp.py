import torch
import torch.nn as nn
import pytorch_lightning as pl


class StaticComp(pl.LightningModule):
    def __init__(self, params, min_db=-80):
        super(StaticComp, self).__init__()
        self.curve_type = params.pop('type')
        self.params = params

        if self.curve_type == 'hk':
            self.model = SimpleHardKnee(**self.params)
        elif self.curve_type == 'sk':
            self.model = SimpleSoftKnee(**self.params)

        self.min_db = min_db
        self.eps = 10 ** (self.min_db / 20)
        self.test_in = torch.linspace(self.min_db, 0, 1000, device=self.device)

    def get_leg(self, cond):
        return self.model.get_leg(cond)

    def forward(self, x, conds=None):
        # Convert to dB/abs
        x = 20 * torch.log10(torch.clamp(torch.abs(x), self.eps))
        # Get static gain curve
        g = self.model(x, conds)
        # Return the gain curve
        return g

    def make_static_in(self):
        return torch.linspace(self.min_db, 0, 1000, device=self.device).unsqueeze(0).unsqueeze(2)

    def get_curve(self, cond_val):
        static_in = self.make_static_in()
        x_ret = self.model(static_in, cond_val.unsqueeze(0))
        x_ret += static_in
        return static_in, x_ret.squeeze()

# A simple two parameter static-compression curve generator
class SimpleHardKnee(pl.LightningModule):
    def __init__(self, cond_size=1, HC_hidden=20, min_db=-80):
        super(SimpleHardKnee, self).__init__()

        self.HCNet = nn.Sequential(
            nn.Linear(in_features=cond_size, out_features=HC_hidden),
            nn.Tanh(),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.Tanh(),
            nn.Linear(in_features=HC_hidden, out_features=2))

        self.min_db = min_db

    def forward(self, x, cond):
        params = self.HCNet(cond)
        threshold, ratio = self.get_pars(params)

        xsc = torch.where(x >= threshold, threshold + ((x - threshold) / ratio), x)
        return xsc - x

    def get_leg(self, cond):
        params = self.HCNet(cond)
        threshold, ratio = self.get_pars(params)

        T = str(-int(torch.round(threshold).item()))
        R = str(min(int(torch.round(ratio).item()), 30))
        T = T + ','
        R = R + ','
        T = T + '  ' if len(T) == 2 else T
        R = R + '  ' if len(R) == 2 else R
        return ' T=' + T + ' R=' + R

    # Function to convert outputs of HCNet to compression curve parameters
    def get_pars(self, params):
        threshold = params[:, 0:1].unsqueeze(1)
        threshold = self.min_db * torch.sigmoid(threshold)

        ratio = params[:, 1:2].unsqueeze(1)
        ratio = 30 * (torch.sigmoid(ratio)) + 1

        return threshold, ratio

class SimpleSoftKnee(pl.LightningModule):
    def __init__(self, cond_size=1, HC_hidden=20,  log_dom=True, min_db=-80):
        super(SimpleSoftKnee, self).__init__()
        self.HCNet = nn.Sequential(
            nn.Linear(in_features=cond_size, out_features=HC_hidden),
            nn.Tanh(),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.Tanh(),
            nn.Linear(in_features=HC_hidden, out_features=3)
        )
        self.log_dom = log_dom
        self.min_db = min_db

    def forward(self, xdb, cond):
        params = self.HCNet(cond)
        threshold, ratio, kw = self.get_pars(params)

        xind = 2*(xdb-threshold)

        out1 = torch.where(xind < -kw, xdb, torch.zeros(1, device=self.device))
        out2 = torch.where(torch.abs(xind) <= kw,
                           xdb + (((1/ratio) - 1) *((xdb - threshold + (kw/2))**2) / (2*kw)), torch.zeros(1, device=self.device))
        out3 = torch.where(xind > kw, threshold + ((xdb - threshold)/ratio), torch.zeros(1, device=self.device))
        return out1 + out2 + out3 - xdb

    def get_leg(self, cond):
        params = self.HCNet(cond)
        threshold, ratio, kw = self.get_pars(params)

        T = str(-int(torch.round(threshold).item()))
        R = str(int(torch.round(ratio).item()))
        W = str(int(torch.round(kw).item()))
        T = T + ','
        R = R + ','
        W = W + ','
        T = T + '  ' if len(T) == 2 else T
        R = R + '  ' if len(R) == 2 else R
        W = W + '  ' if len(W) == 2 else W
        return ' T=' + T + ' R=' + R + ' W=' + W

    # Function to convert outputs of HCNet to compression curve parameters
    def get_pars(self, params):
        threshold = params[:, 0:1].unsqueeze(1)
        threshold = self.min_db * torch.sigmoid(threshold)

        ratio = params[:, 1:2].unsqueeze(1)
        ratio = 30 * (torch.sigmoid(ratio)) + 1

        kw = params[:, 2:3].unsqueeze(1)
        kw = 30*(torch.sigmoid(kw))

        return threshold, ratio, kw
