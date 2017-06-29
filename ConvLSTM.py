import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    #Convolution shape is fixed
    def __init__(self, input_shape, input_channel,filter_size, feature_size):
        super(ConvLSTMCell, self).__init__()
        self.input_shape = input_shape # HxW
        self.input_channel = input_channel
        self.filter_size = filter_size
        self.feature_size = feature_size
        self.padding = (filter_size-1)/2
        self.layer_xi = nn.Conv2d(input_channel,feature_size,filter_size,1,self.padding)
        self.layer_hi = nn.Conv2d(feature_size,feature_size,filter_size,1,self.padding)
        self.layer_xf = nn.Conv2d(input_channel,feature_size,filter_size,1,self.padding)
        self.layer_hf = nn.Conv2d(feature_size,feature_size,filter_size,1,self.padding)
        #self.layer_xc = nn.Conv2d(input_channel,feature_size,filter_size,1,self.padding)
        # self.layer_hc = nn.Conv2d(feature_size,feature_size,filter_size,1,self.padding)
        self.layer_xo = nn.Conv2d(input_channel,feature_size,filter_size,1,self.padding)
        self.layer_ho = nn.Conv2d(feature_size,feature_size,filter_size,1,self.padding)
        #self.register_paramter('w_ci', None)
        #self.register_paramter('w_cf', None)
        # self.register_paramter('w_co', None)
        self.register_parameter('b_i', None)
        self.register_parameter('b_f', None)
        # self.register_paramter('b_c', None)
        self.register_parameter('b_o', None)

    def reset_paramter(self,input):
        #TODO: other option to reset the parameters?
        if self.b_i is None:
            self.b_i = nn.Parameter(input.new((input.size(0), input.size(1), 1)).normal_(0,1))
        elif self.b_f is None:
            self.b_f = nn.Parameter(input.new((input.size(0), input.size(1), 1)).normal_(0,1))
        elif self.b_o is None:
            self.b_o = nn.Parameter(input.new((input.size(0), input.size(1), 1)).normal_(0,1))

    def inti_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size,self.feature_size, self.input_shape[0], self.input_shape[1])).cuda(),
                Variable(torch.zeros(batch_size,self.feature_size, self.input_shape[0], self.input_shape[1])).cuda())

    def add_bias(self, tensor1, bias):
        #tensor1_size: NxCxHxW ; bias: NxCx1
        for batches in xrange(tensor1.size(0)):
            for chans in xrange(tensor1.size(1)):
                tensor1[batches, chans,:,:] = torch.add(tensor1[batches, chans,:,:],
                                                        bias[batches,chans,:])
        return tensor1



    def forward(self,input,hidden):

        hx, cx = hidden

        # Check whether the parameters are set, if not, reset thge parameters
        self.reset_paramter(cx)

        # Calculation of i(t)
        input_i = self.layer_xi(input)
        hx_i    = self.layer_hi(hx)
        i_t1    = torch.add(input_i, hx_i)
        i_t2    = self.add_bias(i_t1, self.b_i)
        i_t     = F.sigmoid(i_t2)

        # Calculation of f(t)
        input_f = self.layer_xf(input)
        hx_f    = self.layer_hf(hx)
        f_t1    = torch.add(input_f, hx_f)
        f_t2    = self.add_bias(f_t1, self.b_f)
        f_t     = F.sigmoid(f_t2)

        # Calculation of c(t)
        c_t1    = torch.mul(f_t, cx)
        input_c = self.layer_xc(input)
        hx_c    = self.layer_hc(hx)
        c_t2    = torch.add(input_c, hx_c)
        c_t3    = self.add_bias(c_t2, self.b_c)
        c_t5    = torch.mul(i_t, F.tanh(c_t3))
        c_t     = torch.addcmul(c_t5, f_t, cx)

        # Calculation of o(t)
        input_o = self.layer_xo(input)
        h_o     = self.layer_ho(hx)
        o_t1    = torch.add(input_o, h_o)
        o_t2    = self.add_bias(o_t1, b_o)
        o_t3    = torch.add(o_t2, o_t3)
        o_t     = F.sigmoid(o_t)

        # Calculation of h(t)
        h_t1    = F.tanh(c_t)
        h_t     = torch.mul(o_t, h_t1)

        return h_t, c_t, o_t

class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_chans, filter_size, feature_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_shape = input_shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.feature_size = feature_size
        self.num_layers  = num_layers

        cell_list = []
        cell_list.append(ConvLSTMCell(self.input_shape, self.input_chans,
                                      self.filter_size, self.feature_size).cuda())

        for idcell in xrange(1,self.num_layers):
            cell_list.append(ConvLSTMCell(self.input_shape, self.feature_size, self.filter_size,
                                          self.feature_size).cuda())
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        current_input = input
        next_hidden   = []
        seq_len       = current_input.size(0)

        for layer in xrange(self.num_layers):

            hidden_c = hidden_state[layer]
            all_output = []
            output_inner = []

            for t in xrange(seq_len):
                hidden_c = self.cell_list[layer](current_input[t,...], hidden_c)
                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner,0).view(current_input.size(0), *output_inner[0].size())

        return next_hidden, current_input

    def inti_hidden(self, batch_size):
        inti_states = []
        for i in xrange(self.num_layers):
            inti_states.append(self.cell_list[i].inti_hidden(batch_size))
        return inti_states

#####usage#####
feature_size = 10
filter_size  = 5
batch_size   = 10
shape = (25,25)
input_channels = 3
nlayers = 1
seq_len = 4

input = Variable(torch.rand(batch_size, seq_len,input_channels, shape[0],shape[1])).cuda()
conv_lstm = ConvLSTM(shape, input_channels, filter_size, feature_size, nlayers)
#conv_lstm.apply(weights_init)
conv_lstm.cuda()

print 'convlstm module: ', conv_lstm
print 'paras: '
paras = conv_lstm.parameters()
for p in paras:
    print 'param ', p.size()
    print 'mean ', torch.mean(p)

hidden_state = conv_lstm.inti_hidden(batch_size)
print 'hidden_h shape ',hidden_state[0][0].size()
out=conv_lstm(input,hidden_state)
print 'out shape',out[1].size()
print 'len hidden ', len(out[0])
print 'next hidden',out[0][0][0].size()
print 'convlstm dict',conv_lstm.state_dict().keys()
