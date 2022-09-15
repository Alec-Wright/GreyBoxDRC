import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


class GainSmooth(pl.LightningModule):
    def __init__(self, params, trunc_steps, min_db=-80):
        super(GainSmooth, self).__init__()
        self.smooth_type = params.pop('type')
        self.params = params

        self.min_db = min_db
        self.eps = 10 ** (self.min_db / 20)
        self.data_min = torch.tensor(self.min_db, device=self.device)
        self.data_range = torch.tensor(-self.min_db, device=self.device)

        self.tvop = False
        if self.smooth_type == 'OnePoleAttOnly':
            self.model = OnePoleAttOnly(**self.params, trunc_steps=trunc_steps)
        if self.smooth_type == 'OnePoleAttRel':
            self.model = OnePoleAttRel(**self.params)
        if self.smooth_type == 'TimeVaryOP':
            self.model = TimeVaryOP(**self.params)
            self.tvop = True

    def forward(self, gains, cond=None):
        gains = self.scale_inputs(gains)
        gains = self.model(gains, cond) if self.model.cond else self.model(gains)
        gains = self.scale_outputs(gains)
        return gains

    def verbose_forward(self, gains, cond=None):
        gains = self.scale_inputs(gains)
        gains, taus = self.model.v_forward(gains, cond) if self.model.cond else self.model.v_forward(gains)
        gains = self.scale_outputs(gains)
        return gains, taus

    def reset_state(self, batch_size):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()

    def make_step_in(self, step_samps=4410):
        return torch.cat((torch.zeros(step_samps, 1, device=self.device),
                          -40*torch.ones(step_samps, 1, device=self.device),
                          torch.zeros(step_samps, 1, device=self.device)))

    def get_step_resp(self, cond=None):
        self.reset_state(1)
        step_samps = 4410*2
        in_data = self.make_step_in(step_samps).unsqueeze(1).permute(1, 0, 2)
        if self.model.cond:
            step_r = self(in_data, cond.unsqueeze(0))
        else:
            step_r = self(in_data)
        return step_r[0, :, 0].squeeze(), in_data[0, :, 0].squeeze()

    def scale_inputs(self, x):
        return x/self.data_range

    def scale_outputs(self, y):
        return y*self.data_range


# One pole filter with attack time = release time, can be implemented in parallel to save computational cost in training
class OnePoleAttOnly(pl.LightningModule):
    def __init__(self, trunc_steps=None, cond=False):
        super(OnePoleAttOnly, self).__init__()
        self.filts = 1
        self.trunc_steps = trunc_steps
        self.cond = cond

        # Create frequency sampling vector for parallel implementation of OP filter
        self.samp_steps = 80000
        self.z_tr = torch.exp(torch.complex(torch.zeros(1), -torch.ones(1)) *
                              torch.linspace(0, math.pi, 1 + self.samp_steps//2))
        self.z_tr = self.z_tr.repeat(1, self.filts, 1).permute(0, 2, 1)

        self.iir_state, self.fir_state = None, None

        if not self.cond:
            self.taus = nn.Parameter(torch.sigmoid(torch.randn(1, 1, self.filts)) * 0.1)
        elif self.cond:
            self.HCNet = nn.Sequential(
                nn.Linear(in_features=1, out_features=20),
                nn.Tanh(),
                nn.Linear(in_features=20, out_features=20),
                nn.Tanh(),
                nn.Linear(in_features=20, out_features=self.filts)
            )

    def forward(self, x, cond=None):
        if self.cond:
            params = self.HCNet(cond)
            taus = 0.5 * torch.sigmoid(params[:, :self.filts]).unsqueeze(1)
        else:
            self.taus.data = self.taus.data.clamp(min=1e-4, max=1)
            taus = self.taus

        alphas = torch.exp(-1 / (taus * 44100))

        # for shorter sequences use the parallel implementation, saves time during backprop
        if x.shape[1] <= 40000:
            filts_out = self.filt_para(alphas, x)
        # for longer sequences (i.e during validation) use recursive, as no backprop is required
        else:
            filts_out = self.filt_recurse(alphas, x)
        return filts_out

    def detach_state(self):
        self.iir_state = self.iir_state.clone().detach()
        self.fir_state = self.fir_state.clone().detach()

    def reset_state(self, batch_size):
        self.iir_state = torch.zeros((batch_size, 1, self.filts), device=self.device)
        self.fir_state = torch.zeros((batch_size, self.samp_steps, self.filts), device=self.device)
        self.z_tr = self.z_tr.to(self.device)

    def filt_recurse(self, alphas, inp):
        inp = inp.repeat(1, 1, alphas.shape[2])
        output = torch.zeros_like(inp)
        for samps in range(inp.shape[1]):
            self.iir_state = torch.mul(alphas, self.iir_state) + torch.mul(1 - alphas, inp[:, samps:samps+1, :])
            output[:, samps:samps+1, :] = self.iir_state
        return torch.mean(output, dim=2, keepdim=True)

    def filt_para(self, alphas, inp):
        # Expand input to have the same number of channels as there are filters, and load the input into the FIR buffer
        inp = inp.repeat(1, 1, alphas.shape[2])
        self.fir_state = torch.cat((self.fir_state[:, inp.shape[1]:, :], inp), dim=1)

        # Calculate the frequency response of the filter/s
        h = (1-alphas)/(1 - alphas*self.z_tr)

        # fft the fir buffer and apply the one-pole filter/s in the frequency domain
        input_fft = torch.fft.rfft(self.fir_state, dim=1)
        output_fft = torch.mul(h, input_fft)

        # Take the ifft to get the filter output, then truncate to get rid of the extra samples from the buffer
        output = torch.fft.irfft(output_fft, n=self.samp_steps, dim=1)
        output = output[:, -inp.shape[1]:, :]
        return torch.mean(output, dim=2, keepdim=True)


# One pole with independent attack and release times, can't be implemented in parallel
class OnePoleAttRel(pl.LightningModule):
    def __init__(self, trunc_steps=None, cond=False):
        super(OnePoleAttRel, self).__init__()
        self.cond = cond
        self.iir_state = None
        self.taus = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)

    def forward(self, x, cond=None):
        taus = torch.sigmoid(self.taus)
        alphas = torch.exp(-1 / (taus * 44100))
        att_alph = alphas[:, :, 0:1]
        rel_alph = alphas[:, :, 1:2]

        gs = torch.zeros_like(x)
        for n in range(x.shape[1]):
            gc = x[:, n:n+1, :]
            self.iir_state = torch.where(self.iir_state > gc,
                                         att_alph * self.iir_state + (1 - att_alph) * gc,
                                         rel_alph * self.iir_state + (1 - rel_alph) * gc)
            gs[:, n:n+1, 0:1] = self.iir_state
        return gs

    def detach_state(self):
        self.iir_state = self.iir_state.clone().detach()

    def reset_state(self, batch_size):
        self.iir_state = torch.zeros((batch_size, 1, 1), device=self.device)


# One pole with time-varying parameter
class TimeVaryOP(pl.LightningModule):
    def __init__(self, hidden_size, rec):
        super(TimeVaryOP, self).__init__()

        if rec == 'gru':
            self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        elif rec == 'rnn':
            self.rec = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)

        self.lin = nn.Linear(in_features=hidden_size, out_features=1)
        self.iir_state = None
        self.state = None
        self.cond = False

    def forward(self, g):
        # Time-varying time constant predicted by recurrent unit
        taus, self.state = self.rec(g, self.state)
        taus = torch.sigmoid(self.lin(taus))
        # Convert time constant to filter parameter alpha
        alphas = torch.exp(-1 / (taus * 44100))

        # Apply one pole filter
        gs = torch.empty_like(g)
        iir_out = self.iir_state
        for n in range(g.shape[1]):
            gc = g[:, n:n+1, :]
            iir_out = alphas[:, n:n+1, :]*iir_out + (1-alphas[:, n:n+1, :])*gc
            gs[:, n:n+1, 0:1] = iir_out
        self.iir_state = iir_out
        return gs

    def verbose_forward(self, g):
        taus, self.state = self.rec(g, self.state)
        taus = torch.sigmoid(self.lin(taus))

        alphas = torch.exp(-1 / (taus * 44100))

        gs = torch.zeros_like(g)
        iir_out = self.iir_state
        for n in range(g.shape[1]):
            gc = g[:, n:n+1, :]
            iir_out = alphas[:, n:n+1, :]*iir_out + (1-alphas[:, n:n+1, :])*gc
            gs[:, n:n+1, 0:1] = iir_out
        self.iir_state = iir_out
        return gs, taus

    def reset_state(self, batch_size=None):
        self.iir_state = torch.zeros((batch_size, 1, 1), device=self.device)
        self.state = None

    def detach_state(self):
        self.iir_state = self.iir_state.clone().detach()
        self.state = self.state.detach()
