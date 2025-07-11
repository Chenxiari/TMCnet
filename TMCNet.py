import torch
from backbone.tans import transnext_base as ts_b
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(int(in_channels),int(in_channels // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels // ratio), int(in_channels), 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM(nn.Module):

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x

# class Transformer(nn.Module):
#     def __init__(self, in_channels):
#         super(Transformer, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = self.in_channels // 2
#         self.bn_relu = nn.Sequential(
#             nn.BatchNorm1d(self.in_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.theta = nn.Linear(self.in_channels, self.inter_channels)
#         self.phi = nn.Linear(self.in_channels, self.inter_channels)
#         self.g = nn.Linear(self.in_channels, self.inter_channels)
#         self.W = nn.Linear(self.inter_channels, self.in_channels)
#
#     def forward(self, ori_feature):
#         shape = ori_feature.shape
#         batch_size = shape[0]
#         C = shape[1]
#         H = shape[2]
#         W = shape[3]
#         ori_feature = ori_feature.view(batch_size,-1,C)
#         ori_feature = ori_feature.permute(0, 2, 1)
#         feature = self.bn_relu(ori_feature)
#         feature = feature.permute(0, 2, 1)
#         N, num, c = feature.size()
#         x_theta = self.theta(feature)
#         x_phi = self.phi(feature)
#         x_phi = x_phi.permute(0, 2, 1)
#         attention = torch.matmul(x_theta, x_phi)
#         f_div_C = F.softmax(attention, dim=-1)
#         g_x = self.g(feature)
#         y = torch.matmul(f_div_C, g_x)
#         W_y = self.W(y).contiguous().view(N, num, c)
#         att_fea = ori_feature.permute(0, 2, 1) + W_y
#         att_fea = att_fea.permute(0,2,1)
#         att_fea = att_fea.view(batch_size,C,H,W)
#
#         return att_fea

class module_1_start(nn.Module):
    def __init__(self,upchannel,downchannel):
        super(module_1_start, self).__init__()
        self.cbam_up = CBAM(in_channels=upchannel)
        self.cbam_down = CBAM(in_channels=upchannel)

        self.s1_up_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=upchannel, out_channels=upchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(upchannel),
            nn.ReLU()
        )

        self.s1_down_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=downchannel, out_channels=upchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(upchannel),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=upchannel * 2, out_channels=upchannel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(upchannel // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=upchannel//2, out_channels=upchannel//2, kernel_size=1, padding=0),
            nn.BatchNorm2d(upchannel//2),
            nn.ReLU(),
        )

        self.fc_1 = nn.Linear(upchannel, upchannel)
        self.fc_2 = nn.Linear(upchannel, upchannel)
        self.fc_3 = nn.Linear(upchannel, upchannel)
        self.flatten = nn.Flatten()


    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self,up,down):
        shape = up.shape
        batch_size = shape[0]
        c = shape[1]
        H = shape[2]
        W = shape[3]

        up = self.s1_up_bn_relu(up)
        up = self.cbam_up(up)
        up_gap = self.gap(up)
        up_gap_reshape = up_gap.view(batch_size, -1)

        down = F.interpolate(down, size=(H, W), mode='bilinear')
        down = self.s1_down_bn_relu(down)
        down = self.cbam_down(down)
        down_gap = self.gap(down)
        down_gap_reshape = down_gap.view(batch_size, -1)


        up_weight = self.fc_1(up_gap_reshape)
        down_weight = self.fc_2(down_gap_reshape)

        weight = up_weight + down_weight
        weight = self.fc_3(weight)

        weight = weight.view(batch_size,-1,1,1)

        up = up * weight + up
        down = down * weight + down

        x = torch.cat([up, down], dim=1)

        x = self.channel_shuffle(x,groups=4)
        x = self.conv(x)

        return x

class module_1(nn.Module):
    def __init__(self,upchannel,mediumchannel,downchannel):
        super(module_1, self).__init__()
        self.cbam_up = CBAM(in_channels=mediumchannel)
        self.cbam_med = CBAM(in_channels=mediumchannel)
        self.cbam_down = CBAM(in_channels=mediumchannel)

        self.s1_up_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=upchannel, out_channels=mediumchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mediumchannel),
            nn.ReLU()
        )

        self.s1_med_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=mediumchannel, out_channels=mediumchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mediumchannel),
            nn.ReLU()
        )

        self.s1_down_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=downchannel, out_channels=mediumchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mediumchannel),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=mediumchannel * 3, out_channels=mediumchannel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mediumchannel // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=mediumchannel//2, out_channels=mediumchannel//2, kernel_size=1, padding=0),
            nn.BatchNorm2d(mediumchannel//2),
            nn.ReLU(),
        )

        self.fc_1 = nn.Linear(mediumchannel, mediumchannel)
        self.fc_2 = nn.Linear(mediumchannel, mediumchannel)
        self.fc_3 = nn.Linear(mediumchannel, mediumchannel)
        self.fc_4 = nn.Linear(mediumchannel, mediumchannel)
        self.flatten = nn.Flatten()

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self,up,med,down):
        shape = med.shape
        batch_size = shape[0]
        H = shape[2]
        W = shape[3]

        up = F.interpolate(up, size=(H, W), mode='bilinear')
        up = self.s1_up_bn_relu(up)
        up = self.cbam_up(up)
        up_gap = self.gap(up)
        up_gap_reshape = up_gap.view(batch_size, -1)


        med = self.s1_med_bn_relu(med)
        med = self.cbam_med(med)
        med_gap = self.gap(med)
        med_gap_reshape = med_gap.view(batch_size, -1)

        down = F.interpolate(down, size=(H, W), mode='bilinear')
        down = self.s1_down_bn_relu(down)
        down = self.cbam_down(down)
        down_gap = self.gap(down)
        down_gap_reshape = down_gap.view(batch_size, -1)


        up_weight = self.fc_1(up_gap_reshape)
        med_weight = self.fc_2(med_gap_reshape)
        down_weight = self.fc_3(down_gap_reshape)


        weight = up_weight + down_weight + med_weight
        weight = self.fc_4(weight)
        weight = weight.view(batch_size, -1, 1, 1)

        up = up * weight + up
        med = med * weight + med
        down = down * weight + down

        x = torch.cat([up,med,down],dim=1)
        x = self.channel_shuffle(x,groups=4)
        x = self.conv(x)
        return x


class module_1_end(nn.Module):
    def __init__(self, upchannel, downchannel):
        super(module_1_end, self).__init__()
        self.cbam_up = CBAM(in_channels=downchannel)
        self.cbam_down = CBAM(in_channels=downchannel)

        self.s1_up_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=upchannel, out_channels=downchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(downchannel),
            nn.ReLU()
        )

        self.s1_down_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=downchannel, out_channels=downchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(downchannel),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=downchannel * 2, out_channels=downchannel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(downchannel // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=downchannel//2, out_channels=downchannel//2, kernel_size=1, padding=0),
            nn.BatchNorm2d(downchannel//2),
            nn.ReLU(),

        )

        self.fc_1 = nn.Linear(downchannel,downchannel)
        self.fc_2 = nn.Linear(downchannel, downchannel)
        self.fc_3 = nn.Linear(downchannel, downchannel)
        self.flatten = nn.Flatten()

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x


    def forward(self, up, down):
        shape = down.shape
        batch_size = shape[0]
        H = shape[2]
        W = shape[3]

        up = F.interpolate(up, size=(H, W), mode='bilinear')
        up = self.s1_up_bn_relu(up)
        up = self.cbam_up(up)
        up_gap = self.gap(up)
        up_gap_reshape = up_gap.view(batch_size, -1)


        down = self.s1_down_bn_relu(down)
        down = self.cbam_down(down)
        down_gap = self.gap(down)
        down_gap_reshape = down_gap.view(batch_size, -1)

        up_weight = self.fc_1(up_gap_reshape)
        down_weight = self.fc_2(down_gap_reshape)

        weight = up_weight + down_weight
        weight = self.fc_3(weight)
        weight = weight.view(batch_size, -1, 1, 1)

        up = up * weight + up
        down = down * weight + down

        x = torch.cat([up, down], dim=1)
        x = self.channel_shuffle(x,groups=4)
        x = self.conv(x)

        return x


class module_2_start(nn.Module):
    def __init__(self, inchannel):
        super(module_2_start, self).__init__()
        self.cbam = CBAM(in_channels=inchannel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.con1x1_in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.convrgb_in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel*2, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con1x1_2in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel * 2, out_channels=inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
        )

        self.con1x1_3in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel * 3, out_channels=inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

        )

        self.sigmoid = nn.Sigmoid()

        self.con_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, rgb, d, s):
        batch = rgb.shape[0]
        H = s.shape[2]
        W = s.shape[3]

        rgb_conv = self.con1x1_in_in(rgb)
        d_conv = self.con1x1_in_in(d)

        rgb_sig = self.sigmoid(rgb_conv)
        d_sig = self.sigmoid(d_conv)

        rgb = rgb_conv * rgb_sig + d_conv * rgb_sig + rgb_conv
        d = d_conv * d_sig + rgb_conv * d_sig + d_conv

        rgb_d = torch.cat([rgb,d],dim=1)
        rgb_d = self.channel_shuffle(rgb_d,groups=4)
        rgb_d = self.convrgb_in_in(rgb_d)
        rgb_d = self.cbam(rgb_d)


        s_concate_1 = self.con_1(s)
        s_concate_2 = self.con_2(s)
        s_concate_3 = self.con_3(s)

        s_concate = torch.cat([s_concate_1, s_concate_2, s_concate_3], dim=1)
        s_concate = self.con1x1_3in_in(s_concate)

        x = torch.cat([s_concate, rgb_d], dim=1)
        x = self.channel_shuffle(x,groups=4)
        x = self.con1x1_2in_in(x)
        return x

class module_2(nn.Module):
    def __init__(self,inchannel):
        super(module_2, self).__init__()
        self.cbam = CBAM(in_channels=inchannel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.con1x1_in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.convrgb_in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel*2, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con1x1_2in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel * 2, out_channels=inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
        )

        self.con1x1_3in_in = nn.Sequential(
            nn.Conv2d(in_channels=inchannel * 3, out_channels=inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

        )
        self.sigmoid = nn.Sigmoid()

        self.con_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1,dilation=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.con_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.channel_half = nn.Sequential(
            nn.Conv2d(in_channels=2*inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self,rgb,d,s,s_back):
        batch = rgb.shape[0]
        H = s.shape[2]
        W = s.shape[3]

        rgb_conv = self.con1x1_in_in(rgb)
        d_conv = self.con1x1_in_in(d)

        rgb_sig = self.sigmoid(rgb_conv)
        d_sig = self.sigmoid(d_conv)

        rgb = rgb_conv * rgb_sig + d_conv * rgb_sig + rgb_conv
        d = d_conv * d_sig + rgb_conv * d_sig + d_conv

        rgb_d = torch.cat([rgb,d],dim=1)
        rgb_d = self.channel_shuffle(rgb_d,groups=4)
        rgb_d = self.convrgb_in_in(rgb_d)
        rgb_d = self.cbam(rgb_d)

        s_back = F.interpolate(s_back, size=(H, W), mode='bilinear')

        s_back = self.channel_half(s_back)

        s_concate = torch.cat([s,s_back],dim=1)
        s_concate = self.con1x1_2in_in(s_concate)

        s_concate_1 = self.con_1(s_concate)
        s_concate_2 = self.con_2(s_concate)
        s_concate_3 = self.con_3(s_concate)

        s_concate = torch.cat([s_concate_1,s_concate_2,s_concate_3],dim=1)
        s_concate = self.con1x1_3in_in(s_concate)

        x = torch.cat([s_concate,rgb_d],dim=1)
        x = self.channel_shuffle(x,groups=4)
        x = self.con1x1_2in_in(x)

        return x


class zwnet(nn.Module):
    def __init__(self):
        super(zwnet, self).__init__()
        # self.rgb_swin = Swin(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
        # self.d_swin = Swin(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])

        self.rgb_ts = ts_b()
        self.d_ts = ts_b()


        self.module1_1 = module_1_start(upchannel=192, downchannel=384)
        self.module1_2 = module_1(upchannel=192,mediumchannel=384,downchannel=768)
        self.module1_3 = module_1(upchannel=384, mediumchannel=768,downchannel=1536)
        self.module1_4 = module_1_end(upchannel=768,downchannel=1536)

        self.module_end_16X = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
        )

        self.module_end_8X = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
        )

        self.module_end_4X = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
        )

        self.module_end_2X = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
        )

        self.module2_1 = module_2_start(inchannel=768)
        self.module2_2 = module_2(inchannel=384)
        self.module2_3 = module_2(inchannel=192)
        self.module2_4 = module_2(inchannel=96)

        self.ca_128 = ChannelAttention(in_channels=96)
        self.ca_256 = ChannelAttention(in_channels=192)
        self.ca_512 = ChannelAttention(in_channels=384)
        self.ca_1024 = ChannelAttention(in_channels=768)

        self.conv_stage_2_1_c = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_2_c = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

        )

        self.conv_stage_2_3_c = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_4_c = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.conv_stage_2_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )


    def forward(self,rgb,d):
        sw_d = self.d_ts(d)
        d_1 = sw_d[0]  # 4倍下采样     96*56*56
        d_2 = sw_d[1]  # 8倍下采样     192*28*28
        d_3 = sw_d[2]  # 16倍下采样    384*14*14
        d_4 = sw_d[3]  # 32倍下采样    768*7*7

        sw_rgb = self.rgb_ts(rgb)
        rgb_1 = sw_rgb[0]  # 4倍下采样     96*56*56
        rgb_2 = sw_rgb[1]  # 8倍下采样     192*28*28
        rgb_3 = sw_rgb[2]  # 16倍下采样    384*14*14
        rgb_4 = sw_rgb[3]  # 32倍下采样    768*7*7


        s1 = torch.cat([d_1, rgb_1], dim=1) #192, 56, 56
        s2 = torch.cat([d_2, rgb_2], dim=1) #384, 28, 28
        s3 = torch.cat([d_3, rgb_3], dim=1) #768, 14, 14
        s4 = torch.cat([d_4, rgb_4], dim=1) #1536, 7, 7

        # print(s1.shape)
        # print(s2.shape)
        # print(s3.shape)
        # print(s4.shape)
        # print(s5.shape)


        stage1_1 = self.module1_1(s1, s2)   # 96, 56, 56
        stage1_2 = self.module1_2(s1, s2, s3)   # 192, 28, 28
        stage1_3 = self.module1_3(s2, s3, s4)   #  384, 14, 14
        stage1_4 = self.module1_4(s3, s4)   # 768, 7, 7


        # print(stage1_1.shape)
        # print(stage1_2.shape)
        # print(stage1_3.shape)
        # print(stage1_4.shape)

        rgb_1 = self.ca_128(rgb_1)
        rgb_2 = self.ca_256(rgb_2)
        rgb_3 = self.ca_512(rgb_3)
        rgb_4 = self.ca_1024(rgb_4)


        d_1 = self.ca_128(d_1)
        d_2 = self.ca_256(d_2)
        d_3 = self.ca_512(d_3)
        d_4 = self.ca_1024(d_4)


        stage2_1 = self.module2_1(rgb_4, d_4, stage1_4)  # 768, 7, 7
        stage2_2 = self.module2_2(rgb_3, d_3, stage1_3, stage2_1)  # 256, 28, 28
        stage2_3 = self.module2_3(rgb_2, d_2, stage1_2, stage2_2)  # 128, 56, 56
        stage2_4 = self.module2_4(rgb_1, d_1, stage1_1, stage2_3)  # 64, 112, 112
        # print(stage2_1.shape)
        # print(stage2_2.shape)
        # print(stage2_3.shape)
        # print(stage2_4.shape)

        stage2_1 = F.interpolate(stage2_1,scale_factor=32,mode='bilinear')
        stage2_2 = F.interpolate(stage2_2, scale_factor=16, mode='bilinear')
        stage2_3 = F.interpolate(stage2_3, scale_factor=8, mode='bilinear')
        stage2_4 = F.interpolate(stage2_4, scale_factor=4, mode='bilinear')

        # print(stage2_1.shape)
        # print(stage2_2.shape)
        # print(stage2_3.shape)
        # print(stage2_4.shape)

        stage2_1 = self.conv_stage_2_1_c(stage2_1)
        stage2_1_1 = self.conv_stage_2_1(stage2_1)
        stage2_1 = stage2_1+stage2_1_1
        out_8 = self.module_end_16X(stage2_1)

        stage2_2 = torch.cat([stage2_1, stage2_2], dim=1)
        stage2_2 = self.conv_stage_2_2_c(stage2_2)
        stage2_2_1 = self.conv_stage_2_2(stage2_2)
        stage2_2 = stage2_2 + stage2_2_1
        out_4 = self.module_end_8X(stage2_2)

        stage2_3 = torch.cat([stage2_2, stage2_3], dim=1)
        stage2_3 = self.conv_stage_2_3_c(stage2_3)
        stage2_3_1 = self.conv_stage_2_3(stage2_3)
        stage2_3 = stage2_3 + stage2_3_1
        out_2 = self.module_end_4X(stage2_3)

        stage2_4 = torch.cat([stage2_3, stage2_4], dim=1)
        stage2_4 = self.conv_stage_2_4_c(stage2_4)
        stage2_4_1 = self.conv_stage_2_4(stage2_4)
        stage2_4 = stage2_4 + stage2_4_1
        out = self.module_end_2X(stage2_4)

        return out,out_2,out_4,out_8
# #

# #
# from torchinfo import summary
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = zwnet().to(device)
# input_size_rgb = (1,3,224,224)
# input_size_d = (1,3,224,224)
# print(summary(model, (input_size_rgb,input_size_d)))
