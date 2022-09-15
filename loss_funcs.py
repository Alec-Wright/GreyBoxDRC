import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal

class DC_PreEmph(nn.Module):
    def __init__(self, R=0.995):
        super(DC_PreEmph, self).__init__()

        t, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=2000)
        ir = ir[0][:, 0]

        self.zPad = len(ir) - 1
        self.pars = torch.flipud(torch.tensor(ir, requires_grad=False, dtype=torch.FloatTensor.dtype)).unsqueeze(0).unsqueeze(0)


    def forward(self, output, target):
        output = output.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), output), dim=2)
        target = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), target), dim=2)
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = nn.functional.conv1d(output, self.pars.type_as(output), bias=None)
        target = nn.functional.conv1d(target, self.pars.type_as(output), bias=None)

        return output.permute(0, 2, 1), target.permute(0, 2, 1)

# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self, dc_pre=True):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001
        self.dc_pre = DC_PreEmph() if dc_pre else None


    def forward(self, output, target):

        if self.dc_pre:
            output, target = self.dc_pre(output, target)

        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss

class RMSLoss(nn.Module):
    def __init__(self, window=50):
        super(RMSLoss, self).__init__()
        self.epsilon = 0.00001
        self.window = window


    def forward(self, output, target):

        output = output.unfold(1, self.window, self.window//4).squeeze(2)
        target = target.unfold(1, self.window, self.window//4).squeeze(2)

        output = torch.pow(output, 2)
        target = torch.pow(target, 2)

        output = torch.mean(output, dim=2)
        target = torch.mean(target, dim=2)

        loss = torch.abs(torch.add(target, -output))

        loss = torch.mean(loss)
        return loss

class RMSMLoss(nn.Module):
    def __init__(self, window_sizes=(8, 16, 32, 64), dc_pre=True):
        super(RMSMLoss, self).__init__()
        self.window_sizes = window_sizes
        self.rms_loss = []
        for size in self.window_sizes:
            self.rms_loss.append(RMSLoss(size))
        self.dc_pre = DC_PreEmph() if dc_pre else None

    def forward(self, output, target):
        if self.dc_pre:
            output, target = self.dc_pre(output, target)
        total_loss = 0

        for item in self.rms_loss:
            total_loss += item(output, target)
        total_loss = total_loss/len(self.rms_loss)
        return total_loss