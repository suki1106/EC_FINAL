import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]

        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)


#
# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * (torch.tanh(F.softplus(x)))
class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, ratio=2, dw_size=3, stride=1, ins=False, mish=False):
        super(GhostModule, self).__init__()
        self.out_ch = out_ch
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)
        if ins:
            if mish:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(in_ch, init_channels, kernel_size, stride, kernel_size // 2, bias=True),
                    nn.InstanceNorm2d(init_channels),
                    # nn.ReLU(inplace=True)
                    Mish()
                )

                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=True),
                    nn.InstanceNorm2d(new_channels),
                    # nn.ReLU(inplace=True)
                    Mish()
                )
            else:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(in_ch, init_channels, kernel_size, stride, kernel_size // 2, bias=True),
                    nn.InstanceNorm2d(init_channels),
                    nn.ReLU(inplace=True)
                )

                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=True),
                    nn.InstanceNorm2d(new_channels),
                    nn.ReLU(inplace=True)
                )
        else:
            if mish:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(in_ch, init_channels, kernel_size, stride, kernel_size // 2, bias=True),
                    # nn.ReLU(inplace=True)
                    Mish()
                )

                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=True),
                    # nn.ReLU(inplace=True)
                    Mish()
                )
            else:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(in_ch, init_channels, kernel_size, stride, kernel_size // 2, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=True),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, mish, pre_act, fac=False, ins=False, bn=False, sep=False, d=1):
        super(ConvBlock, self).__init__()
        self.pre_act = pre_act
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ins = ins
        self.bn = bn
        self.ks = ks
        self.padding = self.ks // 2
        self.mish = mish
        self.sep = sep
        self.fac = fac
        self.d = d

        # self.conv_list = nn.ModuleList()
        self.conv_list = []

        if not self.mish and not self.bn and self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(in_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(in_ch),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(in_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),

                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                    ))
        if not self.mish and self.bn and not self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(in_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(in_ch),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(in_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),

                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                    ))
        if not self.mish and not self.bn and not self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        # nn.InstanceNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        # nn.InstanceNorm2d(out_ch),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.ReLU(True),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.ReLU(True),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        # nn.InstanceNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        nn.ReLU(True),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        # nn.InstanceNorm2d(out_ch),
                        nn.ReLU(True),
                    ))
        if self.mish and not self.bn and self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(in_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(out_ch),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(in_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        nn.InstanceNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.InstanceNorm2d(out_ch),
                        Mish(),
                    ))
        if self.mish and self.bn and not self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(in_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(out_ch),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(in_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.BatchNorm2d(out_ch),
                        Mish(),
                    ))
        if self.mish and not self.bn and not self.ins:
            if self.pre_act:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        # nn.InstanceNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        Mish(),
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        # nn.InstanceNorm2d(out_ch),
                    ))
            else:
                if self.sep:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                        Mish(),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=self.ks, stride=1, groups=out_ch,
                                  padding=self.padding * self.d, dilation=self.d),
                        Mish(),

                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                        # nn.InstanceNorm2d(out_ch),
                    ))
                elif self.fac:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(self.ks, 1), stride=1,
                                  padding=(self.padding * self.d, 0), dilation=(self.d, 1)),
                        Mish(),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, self.ks), stride=1,
                                  padding=(0, self.padding * self.d), dilation=(1, self.d)),
                        Mish(),
                    ))
                else:
                    self.conv_list.append(nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.ks, stride=1,
                                  padding=self.padding * self.d, dilation=self.d),
                        # nn.InstanceNorm2d(out_ch),
                        Mish(),
                    ))

        self.conv = nn.Sequential(*self.conv_list)
        del self.conv_list
        # self.conv = self.conv_list[0]

        pass

    def forward(self, x):
        out = self.conv(x)
        return out


def get_func(func_type, in_channel=16, out_channel=16):
    if func_type == 'conv_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=False, pre_act=False)
    elif func_type == 'conv_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=False, pre_act=False)
    elif func_type == 'conv_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=True, pre_act=False)
    elif func_type == 'conv_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=True, pre_act=False)
    elif func_type == 'ghost_conv_relu_3':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=3, ins=False, mish=False)
    elif func_type == 'ghost_conv_relu_5':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=5, ins=False, mish=False)
    elif func_type == 'ghost_conv_mish_3':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=3, ins=False, mish=True)
    elif func_type == 'ghost_conv_mish_5':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=5, ins=False, mish=True)
    elif func_type == 'fac_conv_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, fac=True, ks=3, mish=False, pre_act=False)
    elif func_type == 'fac_conv_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, fac=True, ks=5, mish=False, pre_act=False)
    elif func_type == 'fac_conv_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, fac=True, ks=3, mish=True, pre_act=False)
    elif func_type == 'fac_conv_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, fac=True, ks=5, mish=True, pre_act=False)
    elif func_type == 'conv_in_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=False, pre_act=False)
    elif func_type == 'conv_in_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=False, pre_act=False)
    elif func_type == 'conv_in_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=False)
    elif func_type == 'conv_in_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=False)
    elif func_type == 'ghost_conv_in_relu_3':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=3, ins=True, mish=False)
    elif func_type == 'ghost_conv_in_relu_5':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=5, ins=True, mish=False)
    elif func_type == 'ghost_conv_in_mish_3':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=3, ins=True, mish=True)
    elif func_type == 'ghost_conv_in_mish_5':
        func = GhostModule(in_ch=in_channel, out_ch=out_channel, dw_size=5, ins=True, mish=True)
    elif func_type == 'fac_conv_in_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, fac=True, ks=3, mish=False, pre_act=False)
    elif func_type == 'fac_conv_in_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, fac=True, ks=5, mish=False, pre_act=False)
    elif func_type == 'fac_conv_in_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, fac=True, ks=3, mish=True, pre_act=False)
    elif func_type == 'fac_conv_in_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, fac=True, ks=5, mish=True, pre_act=False)
    elif func_type == 'conv_bn_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=3, mish=False, pre_act=False)
    elif func_type == 'conv_bn_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=5, mish=False, pre_act=False)
    elif func_type == 'conv_bn_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=3, mish=True, pre_act=False)
    elif func_type == 'conv_bn_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=5, mish=True, pre_act=False)
    elif func_type == 'sep_conv_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=False, pre_act=False, sep=True)
    elif func_type == 'sep_conv_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=False, pre_act=False, sep=True)
    elif func_type == 'sep_conv_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=True, pre_act=False, sep=True)
    elif func_type == 'sep_conv_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=True, pre_act=False, sep=True)
    elif func_type == 'sep_conv_in_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=False, pre_act=False, sep=True)
    elif func_type == 'sep_conv_in_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=False, pre_act=False, sep=True)
    elif func_type == 'sep_conv_in_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=False, sep=True)
    elif func_type == 'sep_conv_in_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=False, sep=True)
    elif func_type == 'p_conv_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=False, pre_act=True)
    elif func_type == 'p_conv_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=False, pre_act=True)
    elif func_type == 'p_conv_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=True, pre_act=True)
    elif func_type == 'p_conv_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=True, pre_act=True)
    elif func_type == 'p_conv_in_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=False, pre_act=True)
    elif func_type == 'p_conv_in_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=False, pre_act=True)
    elif func_type == 'p_conv_in_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True)
    elif func_type == 'p_conv_in_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True)
    elif func_type == 'p_conv_in_mish_3_d_2':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True, d=2)
    elif func_type == 'p_conv_in_mish_5_d_2':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True, d=2)
    elif func_type == 'p_conv_in_mish_3_d_4':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True, d=4)
    elif func_type == 'p_conv_in_mish_5_d_4':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True, d=4)
    elif func_type == 'p_conv_in_mish_3_d_8':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True, d=8)
    elif func_type == 'p_conv_in_mish_5_d_8':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True, d=8)
    elif func_type == 'p_conv_in_mish_3_d_16':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True, d=16)
    elif func_type == 'p_conv_in_mish_5_d_16':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True, d=16)
    elif func_type == 'p_conv_bn_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=3, mish=False, pre_act=True)
    elif func_type == 'p_conv_bn_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=5, mish=False, pre_act=True)
    elif func_type == 'p_conv_bn_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=3, mish=True, pre_act=True)
    elif func_type == 'p_conv_bn_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, bn=True, ks=5, mish=True, pre_act=True)
    elif func_type == 'p_sep_conv_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=False, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=False, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=3, mish=True, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=False, ks=5, mish=True, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_in_relu_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=False, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_in_relu_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=False, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_in_mish_3':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=3, mish=True, pre_act=True, sep=True)
    elif func_type == 'p_sep_conv_in_mish_5':
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, ins=True, ks=5, mish=True, pre_act=True, sep=True)

    else:
        raise NotImplementedError
    return func


if __name__ == '__main__':

    def flatten(input_list):
        output_list = []
        while True:
            if input_list == []:
                break
            for index, value in enumerate(input_list):

                if type(value) == list:
                    input_list = value + input_list[index + 1:]
                    break
                else:
                    output_list.append(value)
                    input_list.pop(index)
                    break
        return output_list
    def check_active(node_num, connect_gene):
        active = [None for _ in range(node_num)]
        node_connect = []
        j = 1
        i = 0
        for _ in range(node_num - 1):
            node_connect.append(connect_gene[i:i + j])
            i = i + j
            j += 1
        for p, node in enumerate(node_connect):
            if p != node_num - 2:
                if sum(node) >= 1:
                    active[p + 1] = True
        for k in range(node_num):
            for node in node_connect:
                if k < len(node) and k != node_num - 1:
                    if node[k] == 1:
                        active[k] = True

                elif k == node_num - 1:
                    if sum(node) >= 1:
                        active[k] = True

        pre_index = [None for _ in range(node_num)]
        for m in range(node_num):
            if active[m]:
                if m == 0:
                    pre_index[m] = [m]
                else:
                    p_index = []
                    if sum(node_connect[m - 1]) == 0:
                        pre_index[m] = [0]
                    else:
                        for index, con in enumerate(node_connect[m - 1]):
                            if con == 1:
                                p_index.append(index + 1)
                        if len(p_index) > 0:
                            pre_index[m] = p_index
        out_index = []
        for t in range(node_num):
            pre_index_ = flatten(pre_index[t + 1:])
            if active[t] and t + 1 not in pre_index_:
                out_index.append(t + 1)
        if sum([1 for act in active if act is not None]) == 0:
            out_index = [0]
        return active, pre_index, out_index


    class UnetBlock(nn.Module):
        def __init__(self, base_ch, active, pre_index, out_index, node_func_type):
            super(UnetBlock, self).__init__()
            self.active = active
            self.pre_index = pre_index
            self.out_index = out_index
            channels = [None for _ in range(len(active))]
            middle_channel = base_ch
            for i in range(len(self.active)):
                if self.active[i]:
                    for j, index in enumerate(self.pre_index[i]):
                        if j == 0 and index == 0:
                            channels[i] = [base_ch, middle_channel]
                        else:
                            channels[i] = [middle_channel, middle_channel]

            self.node_operations = []
            for i in range(len(self.active)):
                if self.active[i]:
                    self.node_operations.append(
                        get_func(node_func_type, in_channel=channels[i][0], out_channel=channels[i][1]))
                else:
                    self.node_operations.append(None)

            self.node_operations = nn.ModuleList(self.node_operations)

            if self.out_index == [0]:
                middle_channel = base_ch

            self.init_conv = get_func(node_func_type, in_channel=base_ch, out_channel=base_ch)
            self.final_conv = get_func(node_func_type, in_channel=middle_channel, out_channel=base_ch)
            self.outputs = [None for _ in range(len(self.active) + 1)]

        def forward(self, x):
            outputs = self.outputs
            x = self.init_conv(x)
            outputs[0] = x
            for i in range(1, len(self.active) + 1):
                if self.active[i - 1]:
                    for j, index in enumerate(self.pre_index[i - 1]):
                        if j == 0:
                            input_t = outputs[index]
                        else:
                            input_t = input_t + outputs[index]
                    outputs[i] = self.node_operations[i - 1](input_t)
            for y, o_index in enumerate(self.out_index):
                if y == 0:
                    out = outputs[o_index]
                else:
                    out = out + outputs[o_index]
            out = self.final_conv(out)
            return out
    import numpy as np

    active, pre_index, out_index = check_active(node_num=5, connect_gene=list(np.random.randint(0, 2, size=[10])),
                                                )

    # model = UnetBlock(base_ch=3, active=active, pre_index=pre_index, out_index=out_index,
    #                   node_func_type='conv_in_mish_3').cuda(0)
    
    # model = get_func('conv_in_mish_3', channel=16)
    x = torch.rand(1, 3, 64, 64).cuda(0)
    y = model(x)
    print(model)
    print(y.shape)
    param = count_param(model)
    print('totoal parameters: %.4fM (%d)' % (param / 1e6, param))
