import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.legacy.nn  as L
from CorrelationLayer import CorrLSTM

class LSTMdisp(nn.Module):
    def __init__(self, input_size, max_displacement, hidden_size):
        super(LSTMdisp, self).__init__()
        self.input_shape = input_size;
        self.max_displacement = max_displacement;
        self.hidden_size = hidden_size

        self.corr   = CorrLSTM(self.input_shape, self.hidden_size, self.max_displacement, 1,1,10).cuda()
        self.conv3d = nn.Conv3d(self.max_displacement+1, 64, (self.hidden_size,1,1),bias=True).cuda()
        self.conv2d = nn.Conv2d(64, 1,1,1,0).cuda()


    def forward(self, input1, input2):
        corr = self.corr(input1, input2)
        transfer = self.conv3d(corr)
        transfer_4d = torch.squeeze(transfer)
        return self.conv2d(transfer_4d).squezze()

#####usage testing#####
def test_usage():
    batch_size = 2
    hidden_size = 20
    max_displacement = 5
    stride_1 = 1
    stride_2 = 1
    pad_shift = 20
    input_size = 15
    shape = (12,12)
    input1 = Variable(torch.rand(batch_size, input_size, shape[0], shape[1])).cuda()
    input2 = Variable(torch.rand(batch_size, input_size, shape[0], shape[1])).cuda()
    LSTM_disp = LSTMdisp(input_size, hidden_size, max_displacement)
    #conv_lstm.apply(weights_init)
    LSTM_disp.cuda()

    print 'convlstm module: ', LSTM_disp
    print 'paras: '
    paras = corr_LSTM.parameters()
    for p in paras:
        print 'param ', p.size()
        print 'mean ', torch.mean(p)


    out=corr_LSTM(input1, input2)
    print 'out shape',  out.size()
    # print 'len hidden ', len(out[0])
    # print 'next hidden',out[0][0][0].size()
    print 'convlstm dict',corr_LSTM.state_dict().keys()
