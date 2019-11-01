import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import math


class SincNet(torch.nn.Module):
    """SincNet module
    Applies the SincNet convolution followed by standard 1D convolution
    SincConv can be replaced with a 1D conv by setting sincnet to False

    Inputs must be of shape: (BATCH, 1, RAW_FEATURES)

    Args:
        sincnet     : True applies the SincConv, False is a standard 1D CNN
        nb_filters  : Number of filters in the SincConv

    Returns:
        ndarray: Audio signal with the range from -1 to 1.
    """

    def __init__(self, sincnet=True, nb_filters=512):
        super(SincNet, self).__init__()

        self.channel   = 1
        self.sincnet   = sincnet
        self.filters   = nb_filters

        # Fixed but can be changed
        self.window_length  = 129
        self.sample_rate    = 16000
        self.min_low_hz     = 50
        self.min_band_hz    = 50
        self.max_pool_size  = 3     # CHANGE TO ADJUST TO THE INPUT SIGNAL DIM. HERE IT IS TUNED FOR 25ms.

        if self.sincnet:
            self.conv1 = SincConv(1, self.filters, self.window_length, sample_rate=self.sample_rate, min_low_hz=self.min_low_hz, min_band_hz=self.min_band_hz)
        else:
            self.conv1 = torch.nn.Conv1d(1, self.filters, self.window_length)

        self.conv2 = torch.nn.Conv1d(self.filters, 256, 3)
        self.conv3 = torch.nn.Conv1d(256, 128, 3)
        self.conv4 = torch.nn.Conv1d(128, 128, 3)

    def forward(self, xs_pad, ilens, **kwargs):
        """
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # Initial Size is (Batch,Seq, Fea)
        # Must be turned into (Batch*Seq,Feat)

        batch = xs_pad.size(0)
        seq   = xs_pad.size(1)

        xs_pad = xs_pad.view(batch*seq,1, -1)


        xs_pad = F.relu(F.max_pool1d(self.conv1(xs_pad), 3))
        xs_pad = F.relu(F.max_pool1d(self.conv2(xs_pad), 3))
        xs_pad = F.relu(F.max_pool1d(self.conv3(xs_pad), 3))
        xs_pad = F.relu(F.max_pool1d(self.conv4(xs_pad), 3))

        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = ilens.tolist()

        # Revert to original shape
        xs_pad = xs_pad.view(xs_pad.size(0),-1).view(batch,seq,-1)

        return xs_pad, ilens, None  # no state in this layer

class Sinc2Conv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):

        super(Sinc2Conv,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        self.a = nn.Parameter(torch.ones(self.out_channels).view(-1, 1))
        torch.nn.init.normal_(self.a, mean=0.0, std=1.0)

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n+1).view(1, -1) / self.sample_rate


    def sinc(self, x):
        # Numerically stable definition
        x_left=x[:,0:int((x.shape[1]-1)/2)]
        y_left=torch.sin(x_left) / x_left
        y_right= torch.flip(y_left,dims=[1])

        sinc=torch.cat([y_left,torch.ones([x.shape[0],1]).to(x.device),y_right],dim=1)


        return sinc

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz /self.sample_rate + torch.abs(self.band_hz_)

        fc        = (high + low) / 2
        bandwith  = high - low

        f_times_b = torch.matmul(bandwith, self.n_)
        f_times_f = torch.matmul(fc, self.n_)

        band_pass = 2 * self.a * torch.pow(self.sinc(np.pi*f_times_b), 2) * torch.cos(2*np.pi*f_times_f)

        #max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        #band_pass = band_pass / max_

        self.filters = (band_pass * self.window_).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)

class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):

        super(SincConv,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n+1).view(1, -1) / self.sample_rate


    def sinc(self, x):
        # Numerically stable definition
        x_left=x[:,0:int((x.shape[1]-1)/2)]
        y_left=torch.sin(x_left) / x_left
        y_right= torch.flip(y_left,dims=[1])

        sinc=torch.cat([y_left,torch.ones([x.shape[0],1]).to(x.device),y_right],dim=1)


        return sinc

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz /self.sample_rate + torch.abs(self.band_hz_)

        f_times_t = torch.matmul(low, self.n_)

        low_pass1 = 2 * low * self.sinc(
            2 * math.pi * f_times_t * self.sample_rate)

        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(
            2 * math.pi * f_times_t * self.sample_rate)

        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_

        self.filters = (band_pass * self.window_).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
