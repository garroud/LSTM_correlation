import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CorrLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, max_displacement, stride_1, stride_2,
               pad_shift):
        super(CorrLSTM, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.pad_shift = pad_shift

        self.LSTMunit = nn.LSTM(input_size, hidden_size, 1, batch_first=True).cuda()
        self.padding1 = nn.ZeroPad2d((max_displacement,0,0,0)).cuda()
        # self.padding2 = L.Padding(0,max_displacement).cuda()

    def init_hidden(self, batch_size ):
        #input: NxC
        # h_0 = input.unsqueeze(0)
        # h_0 = self.padding2.forward(hidden)
        # h   = Variable(h_0).cuda()
        h_0   = Variable(torch.zeros(1,batch_size,self.hidden_size)).cuda()
        c_0   = Variable(torch.zeros(1,batch_size,self.hidden_size)).cuda()
        return (h_0,c_0)

    def find_right(self, input, pos):
        indx = Variable(torch.arange(pos+self.max_displacement, pos, -1).long()).cuda()
        temp = torch.index_select(input, 2, indx)

        return torch.transpose(temp, 1,2)
        # return torch.transpose(temp, 0,1)

    def forward(self, input1, input2):
        #input1: NxCxHxW ; input2: NxCxHxW
        input2_padding = self.padding1(input2)
        height = input1.size(2)
        width  = input1.size(3)
        outputs = []
        for i in xrange(height):
            (h_0 , c_0) = self.init_hidden(input1.size(0))
            for j in xrange(width):
                input = self.find_right(input2_padding[:,:,i,:],j)
                input = torch.cat([input1[:,:,i,j].unsqueeze(1), input],1).contiguous()
                output, (h_n,c_n) = self.LSTMunit(input, (h_0, c_0))
                output = torch.transpose(output,0,1)
                output = output.unsqueeze(3)
                outputs.append(output)
        result_map = torch.cat(outputs, 3).view(input1.size(0),self.max_displacement+1, self.hidden_size,  height, width)
        return result_map

#####usage testing#####
def usage_testing():
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
    corr_LSTM = CorrLSTM(input_size, hidden_size, max_displacement, stride_1, stride_2, pad_shift)
    #conv_lstm.apply(weights_init)
    corr_LSTM.cuda()

    print 'convlstm module: ', corr_LSTM
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
    L = torch.sum(out)
    L.backward()
