import numpy as np
import torch
from torch.nn.parallel import data_parallel

from torch.nn import ModuleDict
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from staticnet import logger as log


class _CorePlusReadoutBase(nn.Module):
    def __init__(self, core, readout, modulator=None, nonlinearity=None, shifter=None):
        super().__init__()

        self.core = core
        self.readout = readout
        self.modulator = modulator
        self.shifter = shifter
        self.nonlinearity = torch.nn.functional.softplus if nonlinearity is None else nonlinearity

        self._shift = shifter is not None
        self._modulate = modulator is not None
        self.readout_gpu = None

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, val):
        self._shift = val and self.shifter is not None

    @property
    def modulate(self):
        return self._modulate

    @modulate.setter
    def modulate(self, val):
        self._modulate = val and self.modulator is not None

    @staticmethod
    def get_readout_in_shape(core, in_shape):
        mov_shape = in_shape[1:]
        core.eval()
        tmp = Variable(torch.from_numpy(np.random.randn(1, *mov_shape).astype(np.float32)))
        nout = core(tmp).size()[1:]
        core.train(True)
        return nout


class CorePlusReadout2d(_CorePlusReadoutBase):
    
    @property
    def state(self):
        return dict(shift=self.shift, modulate=self.modulate)

    def forward(self, x, behavior=None):
        # 使用 readout 的第一个 key 作为默认值
        readout_key = next(iter(self.readout.keys()))

        if getattr(self.core, 'multiple_outputs', False):
            # if self.core outputs key specific outputs
            x = self.core(x, readout_key)
        else:
            x = self.core(x)

        # 移除 eye_pos 处理，直接使用默认的 readout
        x = self.readout[readout_key](x)

        if behavior is not None and self.modulator is not None and self.modulate:
            x = self.modulator[readout_key](behavior, x)
        return self.nonlinearity(x)

    def neuron_layer_power(self, x, readout_key, neuron_id):
        x = self.core(x)
        return self.readout[readout_key].neuron_layer_power(x, neuron_id)
