import torch
from torch import nn
from utils import idx2onehot
from image_utils import onehot2label,label2onehot
from collections import OrderedDict
import torch.nn.functional as F
import consts
# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, D, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.D, self.H, self.W = C, D, H, W

    def forward(self, input):
        return input.view(input.size(0), self.C, self.D, self.H, self.W)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class Encoder_caae(nn.Module):
    def __init__(self):
        super(Encoder_caae, self).__init__()

        num_conv_layers = 6

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, padding, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    act_fn
                )
            )
        kernel_size = [5, 5, 5]
        padding_size = 1
        add_conv(self.conv_layers, 'e_conv_1', in_ch=8, out_ch=64, kernel=kernel_size, stride=2, padding=padding_size, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', in_ch=64, out_ch=128, kernel=kernel_size, stride=2, padding=padding_size, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', in_ch=128, out_ch=256, kernel=kernel_size, stride=2, padding=padding_size, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', in_ch=256, out_ch=512, kernel=kernel_size, stride=2, padding=padding_size, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', in_ch=512, out_ch=1024, kernel=[1,5,5], stride=2, padding=padding_size, act_fn=nn.ReLU())

        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ('e_fc_1', nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS)),
                    ('tanh_1', nn.Tanh())  # normalize to [-1, 1] range
                ]
            )
        )

    def forward(self, seg):
        out = seg
        for conv_layer in self.conv_layers:
            #print("H")
            out = conv_layer(out)
            #print(out.shape)
            #print("W")
        out = out.flatten(1, -1)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ_caae(nn.Module):
    def __init__(self):
        super(DiscriminatorZ_caae, self).__init__()
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dz_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dz_fc_%d' % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            )
        )

    def forward(self, out):
        for layer in self.layers:
            out = layer(out)
        return out


class DiscriminatorImg_caae(nn.Module):
    def __init__(self):
        super(DiscriminatorImg_caae, self).__init__()
        in_dims = (8, 16 + consts.LABEL_LEN_EXPANDED, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm3d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                nn.Linear(128 * 8 * 8 * 4, 1024),
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            )
        )

    def forward(self, out, labels, device):
        # run convs
        for i, conv_layer in enumerate(self.conv_layers, 1):
            # print(out.shape)
            # print(conv_layer)
            out = conv_layer(out)
            if i == 1:
                # concat labels after first conv
                labels_tensor = labels.repeat(out.shape[-3], out.shape[-1], out.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
                out = torch.cat((out, labels_tensor), 1)

        # run fcs
        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            # print(out.shape)
            # print(fc_layer)

            out = fc_layer(out)

        return out


class Generator_caae(nn.Module):
    def __init__(self, args):
        super(Generator_caae, self).__init__()
        num_deconv_layers = 5
        mini_size = 4
        if args.mapping:
            self.fc = nn.Sequential(
                nn.Linear(
                    consts.NUM_Z_CHANNELS + consts.NUM_STYLE,
                    consts.NUM_GEN_CHANNELS * 2 * mini_size ** 2
                ),
                nn.ReLU()
            )
            # self.mapping = MLP(consts.LABEL_LEN_EXPANDED, consts.NUM_STYLE, 64, 8, weight_norm=True)

        else:
            self.fc = nn.Sequential(
                nn.Linear(
                    consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED,
                    consts.NUM_GEN_CHANNELS * 2* mini_size ** 2
                ),
                nn.ReLU()
            )

        # need to reshape now to ?,1024,8,8
        self.deconv_layers= nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 8, 4, 2, 1),
            nn.Tanh(),

        )
        self.mapping = MLP(consts.LABEL_LEN_EXPANDED, consts.NUM_STYLE, 64, 8, weight_norm=True)

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 2, 4, 4)  # TODO - replace hardcoded

    def forward(self, z, args, age=None, gender=None):
        out = z
        if age is not None and gender is not None:
            label = torch.cat((age, gender), 1)
            out = torch.cat((out, label), 1)# z_l
        if args.mapping:
            label = self.mapping(out[:,consts.NUM_Z_CHANNELS:,...])
            out = torch.cat((out[:,:consts.NUM_Z_CHANNELS,...], label), 1)

        out = self.fc(out)
        out = self._decompress(out)
        out = self.deconv_layers(out)

        return out


def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    c_in, h_in, w_in , w2_in= in_dims
    c_out, h_out, w_out, w2_out = out_dims

    padding = [0, 0, 0]
    output_padding = [0, 0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1] + kernel[2]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
        padding[2] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1
        padding[2] = lhs_1 // 2 + 1
        output_padding[2] = 1

    return torch.nn.ConvTranspose3d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )

class AgeClassifier(nn.Module):

    def __init__(self, age_group=7, conv_dim=64, repeat_num=4):
        super(AgeClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(8, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(conv_dim),
            nn.ReLU(True),
        )

        nf_mult = 1
        age_classifier = []
        for n in range(1, repeat_num + 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 64)
            age_classifier += [
                nn.Conv3d(conv_dim * nf_mult_prev,
                          conv_dim * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(conv_dim * nf_mult),
                nn.ReLU(True),
            ]
        self.age_classifier = nn.Sequential(*age_classifier)


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim * nf_mult *4, age_group),#` * 16
        )

    def forward(self, inputs):
        c1 = self.conv1(inputs)
        c2 = self.age_classifier(c1)
        age_logit = self.fc(c2)
        return age_logit

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='relu', normalize_mlp=True):#, pixel_norm=False):
        super(MLP, self).__init__()
        # if weight_norm:
        #     linear = EqualLinear
        # else:
        #     linear = nn.Linear
        linear = nn.Linear
        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        # elif activation == 'blrelu':
        #     actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)
