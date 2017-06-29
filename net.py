import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self._input_shape = input_shape;
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.LeakyRelu(0.1)
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(64,114,5,2,3),
            nn.LeakyRelu(0.1)
        )
        self.norm1 = nn.Batchnormalization(_input_shape[2]*_input_shape[3]/16,
                                           eps=1e-9, affine=False )
        self.corr = CorrelationLayer1D()
        self.layer2 = nn.Sequential(
            nn.Conv2d(114,60,1,1,0),
            nn.LeakyRelu(0.1)
        )
        #input_size = shift_size + 60
        self.layer3 = nn.Sequential(
            nn.Conv2d(100,239,5,2,2)
            nn.LeakyRelu(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(239,244,3,1,1)
            nn.LeakyRelu(0.1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(244,324,3,2,1)
            nn.LeakyRelu(0.1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(324,397,3,1,1)
            nn.LeakyRelu(0.1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(397,250,3,2,1)
            nn.LeakyRelu(0.1)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(250,230,3,1,1)
            nn.LeakyRelu(0.1)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(230,35,3,2,1)
            nn.LeakyRelu(0.1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(35,15,3,1,1)
            nn.LeakyRelu(0.1)
        )
        self.prdflow6 = nn.Sequential(
            nn.Conv2d(15,1,3,1,1)
            nn.Relu()
        )
        self.layer11 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor = 2)
            nn.Conv2d(15,512,3,1,1)
            nn.LeakyRelu(0.1)
        )
        self.layer12 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor = 2)
            nn.Conv2d(1,1,3,1,1)
        )
        #input_size = 230 + 512 + 1
        self.layer13 = nn.Conv2d(743, 512,3,1,1)
        self.layer14 = nn.Conv2d(512,1, 3,1,1)


    def forward(self,left,right):
        # Data augmentation
        left = torch.add(0.00392156862745, left)
        right = torch.add(0.00392156862745, right)
        out2x = self.layer1_1(left)
        out1x = self.layer1_2(out2x)

        out1_left = out1x.permute(0,3,2,1)
        out1_left = out1_left.view(out1_left.size(0), out1_left.size(1) , -1 , 1)
        out1_left = self.norm1(out1_left)
        out1_left = out1_left.view(out1_left.size(0),out1_left.size(1),
                    _input_shape[2]/4, -1)
        out1_left = out1_left.permute(0,3,2,1)

        out_right = self.layer1(right)
        out1_right = out_right.permute(0,3,2,1)
        out1_right = out1_right.view(out1_right.size(0), out1_right.size(1) , -1 , 1)
        out1_right = self.norm1(out1_right)
        out1_right = out1_right.view(out1_right.size(0), out1_right.size(1),
                    _input_shape[2]/4, -1)
        out1_right = out1_right.permute(0,3,2,1)

        out_1 = self.corr(out1_left, out1_right)
        out_2 = self.layer2(out_left)

        out  = torch.cat((out1,out2),0)
        out2 = self.layer3(out)
        out2 = self.layer4(out2)
        out4 = self.layer5(out2)
        out4 = self.layer6(out4)
        out8 = self.layer7(out4)
        out8 = self.layer8(out8)
        out16 = self.layer9(out8)
        out16 = self.layer10(out16)

        #predict_flow6
        pred_flow6 = self.prdflow6(out16)

        out8_2 = self.layer11(out16)
        out8_3 = self.layer12(pred_flow6)
        out8_4 = torch.cat((out8,out8_2,out8_3),0)
        out8_4 = self.layer13(out8_4)

        #predict_flow5
        pred_flow5 = self.layer14(out8_4)
        add_info6  = F.upsample_bilinear(pred_flow6,scale_factor= 2 )
        pred_flow5 = torch.add(pred_flow5, add_info6)
        pred_flow5 = F.relu(pred_flow5)


        out4_2 = self.layer15(out8_2)
        out4_3 = self.layer16(pred_flow5)
        out4_4 = torch.cat((out4,out4_2,out4_3),0)
        out4_4 = self.layer17(out4_4)

        #predict_flow4
        pred_flow4 = self.layer18(out4_4)
        add_info5  = F.upsample_bilinear(pred_flow4,scale_factor=2)
        pred_flow4 = torch.add(pred_flow4, add_info5)
        pred_flow4 = F.relu(pred_flow4)

        out2_2 = self.layer19(out4_2)
        out2_3 = self.layer20(pred_flow4)
        out2_4 = torch.cat((out2,out2_2,out2_3),0)
        out2_4 = self.layer21(out2_4)

        #predict_flow3
        pred_flow3 = self.layer22(out2_4)
        add_info4 = F.upsample_bilinear(pred_flow4,scale_factor=2)
        pred_flow3 = torch.add(pred_flow3,add_info4)
        pred_flow3 = F.relu(pred_flow3)

        out1_2 = self.layer23(out2_2)
        out1_3 = self.layer24(pred_flow3)
        out1_4 = torch.cat((out1x,out1_2,out1_3),0)
        out1_4 = self.layer25(out1_4)

        #predict_flow2
        pred_flow2 = self.layer26(out1_4)
        add_info3  = F.upsample_bilinear(pred_flow3,scale_factor=2)
        pred_flow2 = torch.add(pred_flow2, add_info3)
        pred_flow2 = F.relu(pred_flow2)

        out2x_2 = self.layer27(out1_2)
        out2x_3 = self.layer28(pred_flow2)
        out2x_4 = torch.cat((out2x, out2x_2, out2x_3),0)
        out2x_4 = self.layer29(out2x_4)

        #predict_flow1
        pred_flow1 = self.layer30(out)
        add_info2  = F.upsample_bilinear(pred_flow2, scale_factor = 2)
        pred_flow1 = torch.add(pred_flow1, add_info2)
        pred_flow1 = F.relu(pred_flow1)
