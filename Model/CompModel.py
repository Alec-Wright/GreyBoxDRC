import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import loss_funcs
from DataFuncs import ConditionedDataLoader

from Model.StaticComp import StaticComp
from Model.GainSmooth import GainSmooth
from Model.MakeUp import MakeUp

from scipy.io.wavfile import write
import math
import copy


class CompModel(pl.LightningModule):
    def __init__(self, loss_func, static_comp, gain_smooth, make_up,
                 trunc_steps=8192, warmup=3754, save_size=3, save_clips=[0, 12, 20], data_set='LA-2A', cond_name=[],
                 l_rate=0.005):
        super(CompModel, self).__init__()
        self.l_rate = l_rate
        self.cond_names = cond_name

        # Which outputs to save to tensorboard
        self.save_size = save_size
        self.save_clips = save_clips
        self.data_set = data_set

        # Training update pars
        self.truncated_bptt_steps = trunc_steps
        self.warmup = warmup
        self.test_counter = None
        self.save_hyperparameters()

        # Create the static-compressor curve object
        self.static_comp = StaticComp(static_comp)

        # Create the gain smoothing object
        self.gain_smooth = GainSmooth(gain_smooth, trunc_steps)

        # Create the make-up gain object
        self.make_up = MakeUp(make_up)

        self.loss_functions = {'ESR': loss_funcs.ESRLoss(), 'MAE': torch.nn.L1Loss(), 'rmsm': loss_funcs.RMSMLoss()}

        self.lf = loss_func
        self.loss_fn = self.loss_functions[self.lf]
        self.loss_functions.pop(self.lf)

    def configure_optimizers(self, l_rate=0.005):
        return torch.optim.Adam(self.parameters(), lr=self.l_rate)

    def reset_states(self, batch_size):
        self.gain_smooth.reset_state(batch_size)
        self.make_up.reset_state(batch_size)

    def detach_states(self):
        self.gain_smooth.detach_state()
        self.make_up.detach_state()

    # input x = [batch, length, channels],
    # hiddens are not used, as they maintained internally by the components with states
    def forward(self, x, hiddens, cond1=None, cond2=None):
        # compute static gain characteristic, gc
        gc = self.static_comp(x, cond1)
        # create smoothed gain, gs
        gs = self.gain_smooth(gc, cond2)
        # apply gain envelope to input signal
        x_comp = torch.mul(x, 10**(gs/20))
        # apply make up gain
        x_mkup = self.make_up(x_comp)
        # return the compressed signal, as well as the gain envelope applied
        return x_mkup, gs

    # just for plotting purposes -
    # version of the forward function which returns the time-varying time-constant of the gain smoother, for when the
    # time-varying one-pole filter is used
    def tvop_verbose_forward(self, x, hiddens, cond1=None, cond2=None):
        gc = self.static_comp(x, cond1)
        gs, taus = self.gain_smooth.tvop_verbose_forward(gc, cond2)
        x_comp = torch.mul(x, 10**(gs/20))
        x_mkup = self.make_up(x_comp)
        return x_mkup, gs, taus

    # just for plotting purposes -
    # version of the forward function which returns the static and smoothed gain envelope, and signal before make-up
    def verbose_forward(self, x, hiddens, cond1=None, cond2=None):
        gc = self.static_comp(x, cond1)
        gs = self.gain_smooth(gc, cond2)
        x_comp = torch.mul(x, 10**(gs/20))
        x_mkup = self.make_up(x_comp)
        return x, gc, gs, x_comp, x_mkup

    def tbptt_split_batch(self, batch, split_size):
        return ConditionedDataLoader.tbptt_split_batch(batch, self.warmup, split_size)

    def training_step(self, batch, batch_idx, hiddens=None):
        warmup_step = hiddens is None
        x, y, conds = batch

        if warmup_step:
            # construct dummy loss so that the warmup step does not update parameters
            self.reset_states(batch[0].shape[0])
            loss = torch.zeros(1, device=self.device, requires_grad=True)
            x, y, conds = batch
        else:
            self.detach_states()

        y_hat, hiddens = self(x, hiddens, conds)

        if not warmup_step:
            # in all other steps, calculate actual loss and backpropagate
            loss = self.loss_fn(y_hat, y)
            self.log("train_loss_" + self.lf, loss, on_step=True, on_epoch=True)
        return {"loss": loss, "hiddens": hiddens}

    # Validation step processes each conditioning value in turn, so first step will include all data for PR=0, second
    # for all data for PR=10, etc
    def validation_step(self, batch, batch_idx):
        cond_name = 'Peak_Reduction: ' + self.cond_names[batch_idx]

        with torch.no_grad():
            self.make_plots(self.global_step, batch[2], cond_name)
            self.reset_states(batch[0].shape[0])

            x, y, conds = batch

            y_hat = torch.empty_like(y)
            gs = torch.empty_like(y)
            taus = torch.empty_like(y)

            # To increase processing speed, longer segment is processed in shorter sub sequences
            chunk_s = 44100
            for n in range(5):
                st_ch = n * chunk_s
                en_ch = (n + 1) * chunk_s
                # if time-vary one-pole is used, use function that returns the time-vary one-pole par for plotting
                if self.gain_smooth.tvop:
                    y_hat[:, st_ch:en_ch, :], gs[:, st_ch:en_ch, :], taus[:, st_ch:en_ch, :] = \
                        self.gain_smooth.tvop_verbose_forward(x[:, st_ch:en_ch, :], None, conds)
                else:
                    y_hat[:, st_ch:en_ch, :], gs[:, st_ch:en_ch, :] = self(x[:, st_ch:en_ch, :], None, conds)

            # Calculate and log val loss of function used for training
            act_loss = self.loss_fn(y, y_hat)
            self.log("val_loss_st" + self.lf, act_loss, on_step=True, on_epoch=False)
            self.log("val_loss_" + self.lf, act_loss, on_epoch=True, on_step=False)

            # Calculate and log other loss metrics
            losses = self.calc_vlosses(y, y_hat)
            for loss in losses.keys():
                self.log("val_loss_st" + loss, losses[loss], on_step=True, on_epoch=False)
                self.log("val_loss_ep" + loss, losses[loss], on_epoch=True, on_step=False)

            # Save outputs of the compressor model to tensorboard
            for n in self.save_clips:
                clip_suf = 'Clip' + str(n) + cond_name
                clip_st = int(0.5*44100)
                clip_en = self.save_size * 44100 + clip_st
                if self.current_epoch == 0:
                    self.logger.sub_logger.add_audio(clip_suf + ' -Input', x[n, clip_st:clip_en, 0].cpu().numpy(), 0)
                    self.logger.sub_logger.add_audio(clip_suf + ' -Target', y[n, clip_st:clip_en, 0].cpu().numpy(), 0)
                self.logger.sub_logger.add_audio(clip_suf + ' -Output', y_hat[n, clip_st:clip_en, 0].cpu().numpy(),
                                                 self.global_step)

                # plot the corresponding gain envelopes applied to the input signal
                if self.gain_smooth.tvop:
                    f = make_gain_plot_taus(gain_pred=gs[n, clip_st:clip_en, 0], input=x[n, clip_st:clip_en, 0],
                                            taus=taus[n, clip_st:clip_en, 0])
                else:
                    f = make_gain_plot(gain_pred=gs[n, clip_st:clip_en, 0], input=x[n, clip_st:clip_en, 0])
                self.logger.sub_logger.add_figure(clip_suf + ' Gains:' + cond_name, f, global_step=self.global_step,
                                                  close=True, walltime=None)

    def test_step(self, batch, batch_idx):
        cond_name = 'Peak_Reduction: ' + self.cond_names[batch_idx]
        with torch.no_grad():
            self.make_test_plots(batch_idx, batch[2])
            self.reset_states(batch[0].shape[0])

            x, y, conds = batch

            y_hat, hiddens = self(x, None, conds)

            act_loss = self.loss_fn(y, y_hat)
            self.log("test_loss_" + self.lf, act_loss, 0, on_epoch=True)
            self.log("test_loss_cond_wise" + self.lf, act_loss, 0, on_step=True)

            losses = self.calc_vlosses(y, y_hat)
            for loss in losses.keys():
                self.log("test_loss" + loss, losses[loss], 0, on_epoch=True)
                self.log("test_loss_cond_wise" + loss, losses[loss], 0, on_step=True)

            for n in self.save_clips:
                clip_suf = 'Test_Clip' + str(n)
                clip_st = int(0.5*44100)
                clip_en = 9 * 44100 + clip_st

                self.logger.sub_logger.add_audio(clip_suf + ' -Input', x[n, clip_st:clip_en, 0].cpu().numpy(), batch_idx)
                self.logger.sub_logger.add_audio(clip_suf + ' -Target', y[n, clip_st:clip_en, 0].cpu().numpy(), batch_idx)
                self.logger.sub_logger.add_audio(clip_suf + ' -Output', y_hat[n, clip_st:clip_en, 0].cpu().numpy(),
                                                 batch_idx)

    def calc_vlosses(self, y, y_hat):
        losses = {}
        for funcs in self.loss_functions.keys():
            loss = self.loss_functions[funcs](y, y_hat)
            losses[funcs] = loss
        return losses

    # Method that makes plots of the various components of the compressor model
    def make_plots(self, step, cond, cond_name):
        cond = cond[0]
        f = make_static_plot(*self.static_comp.get_curve(cond))
        self.logger.sub_logger.add_figure('Static_Gain:' + cond_name,
                                          f, global_step=step, close=True, walltime=None)
        f = make_stepr_plot(*self.gain_smooth.get_step_resp(cond))
        plot_name = 'Step_Resp: ' + cond_name if self.gain_smooth.model.cond else 'Step_Resp'
        self.logger.sub_logger.add_figure(plot_name, f, global_step=step, close=True, walltime=None)

    # Method that makes plots of the various components of the compressor model, for use during test step
    def make_test_plots(self, step, cond):
        cond = cond[0]
        f = make_static_plot(*self.static_comp.get_curve(cond))
        self.logger.sub_logger.add_figure('Test_Static_Gain - step represents Peak_Reduction',
                                          f, global_step=step, close=True, walltime=None)
        f = make_stepr_plot(*self.gain_smooth.get_step_resp(cond))
        plot_name = 'Test_Step_Resp'
        self.logger.sub_logger.add_figure(plot_name, f, global_step=step, close=True, walltime=None)


def make_static_plot(db_in, db_pred_out):
    db_in = db_in.cpu().squeeze()
    db_pred_out = db_pred_out.cpu().squeeze()
    f = plt.figure()
    plt.plot(db_in, db_in, label='No Compression')
    plt.plot(db_in, db_pred_out, label='Model Compression', linestyle='dashed')
    plt.title('static compression characteristic')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend()
    return f

def make_stepr_plot(out_sig, in_sig):
    out_sig = out_sig.cpu().squeeze()
    in_sig = in_sig.cpu().squeeze()
    f = plt.figure()
    tAx = torch.linspace(0, len(in_sig)/44100, len(in_sig))
    plt.plot(tAx, in_sig, label='Input')
    plt.plot(tAx, out_sig, label='Output', linestyle='dashed')
    plt.title('step response')
    plt.ylabel('signal level')
    plt.xlabel('time (ms)')
    plt.legend()
    return f

def make_gain_plot(gain_pred, input):
    gain_pred = gain_pred.cpu()
    input = input.cpu()
    clip_len_samps = len(gain_pred)
    clip_len_secs = clip_len_samps//44100
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
    ax2.plot(gain_pred)
    ax1.plot(input)
    ax2.set_xticks(range(0, len(gain_pred) + 1, 44100))
    ax2.set_xticklabels([str(n) for n in range(0, clip_len_secs + 1)])
    ax2.set_ylabel('db gain')
    ax1.set_ylabel('input signal')
    ax2.set_xlabel('time')
    return f

def make_gain_plot_taus(gain_pred, input, taus):
    gain_pred = gain_pred.cpu()
    taus = taus.cpu()
    input = input.cpu()
    clip_len_samps = len(gain_pred)
    clip_len_secs = clip_len_samps//44100
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False)
    ax1.plot(input)
    ax2.plot(gain_pred)
    ax3.plot(taus)

    ax3.set_xticks(range(0, len(gain_pred) + 1, 44100))
    ax3.set_xticklabels([str(n) for n in range(0, clip_len_secs + 1)])

    ax1.set_ylabel('input signal')
    ax2.set_ylabel('db gain')
    ax3.set_ylabel('OP time constant')

    ax3.set_xlabel('time')

    return f