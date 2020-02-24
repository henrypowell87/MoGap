import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable


def clip_grad(nabla, min, max):
    nabla_tmp = nabla.expand_as(nabla)
    nabla_tmp.register_hook(lambda g: g.clamp(min, max))
    return nabla_tmp


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', {nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class IRNNCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=True):
        super(IRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip

        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))

        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias) + h.cuda() * self.weight_hh
        if self.grad_clip:
            output = clip_grad(output, -self.grad_clip, self.grad_clip)
        output = F.relu(output)

        return output


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size, recurrent_size=None, num_layers=1, bias=True,
                 return_sequences=True, grad_clip=None):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        mode2cell = {'IRNN': IRNNCell}
        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'bias': bias,
                  'grad_clip': grad_clip}

        self.cell0 = Cell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = recurrent_size if self.mode == 'LSTMP' else hidden_size
            cell = Cell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        input = input.to(torch.device('cuda:0'))
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            if self.mode == 'LSTM' or self.mode == 'LSTMON':
                initial_states = [(zeros, zeros), ] * self.num_layers
            elif self.mode == 'LSTMP':
                zeros_h = Variable(torch.zeros(input.size(0), self.recurrent_size))
                initial_states = [(zeros_h, zeros), ] * self.num_layers
            else:
                initial_states = [zeros] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx
                if self.mode.startswith('LSTM'):
                    x = hx[0]
                else:
                    x = hx
            outputs.append(hx)

        if self.return_sequences:
            if self.mode.startswith('LSTM'):
                hs, cs = zip(*outputs)
                h = torch.stack(hs).transpose(0, 1)
                c = torch.stack(cs).transpose(0, 1)
                output = (h, c)
            else:
                output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output


class IRNN(RNNBase):

    def __init__(self, *args, **kwargs):
        super(IRNN, self).__init__('IRNN', *args, **kwargs)


class EncoderIRNN(Module):
    def __init__(self, input_size, num_layers, grad_clip=None):
        super(EncoderIRNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.grad_clip = grad_clip

        self.irnn_1e = IRNN(input_size=input_size, hidden_size=1024, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_2e = IRNN(input_size=1024, hidden_size=512, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_3e = IRNN(input_size=512, hidden_size=252, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_4e = IRNN(input_size=252, hidden_size=125, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_5e = IRNN(input_size=125, hidden_size=62, num_layers=num_layers, grad_clip=grad_clip)

    def forward(self, x):
        x = self.irnn_1e(x)
        x = self.irnn_2e(x)
        x = self.irnn_3e(x)
        x = self.irnn_4e(x)
        x = self.irnn_5e(x)

        return x


class DecoderIRNN(Module):
    def __init__(self, num_layers, grad_clip=None):
        super(DecoderIRNN, self).__init__()
        self.num_layers = num_layers
        self.grad_clip = grad_clip

        self.irnn_1d = IRNN(input_size=62, hidden_size=125, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_2d = IRNN(input_size=125, hidden_size=252, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_3d = IRNN(input_size=252, hidden_size=512, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_4d = IRNN(input_size=512, hidden_size=1024, num_layers=num_layers, grad_clip=grad_clip)
        self.irnn_5d = IRNN(input_size=1024, hidden_size=123, num_layers=num_layers, grad_clip=grad_clip)

    def forward(self, x):

        x = self.irnn_1d(x)
        x = self.irnn_2d(x)
        x = self.irnn_3d(x)
        x = self.irnn_4d(x)
        x = self.irnn_5d(x)

        return x


class IRNNAE(Module):
    def __init__(self, input_size, num_layers, grad_clip=None):
        super(IRNNAE, self).__init__()
        self.encoder = EncoderIRNN(input_size=input_size, num_layers=num_layers, grad_clip=grad_clip)
        self.decoder = DecoderIRNN(num_layers=num_layers, grad_clip=grad_clip)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_input = self.decoder(encoded_input)

        return decoded_input