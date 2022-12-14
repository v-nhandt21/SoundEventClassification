from re import M



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchaudio
from stft import TacotronSTFT
import numpy as np


class MelFrontend(TacotronSTFT):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)

    def forward(self, y, random_mask=True, device="cuda"):
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_basic = self.mel_basis.to(device)
        mel_output = torch.matmul(mel_basic, magnitudes.to(device))
        mel_output = self.spectral_normalize(mel_output)
        if random_mask:
            mel_output = self.spectrum_masking(mel_output)
        return mel_output

    @staticmethod
    def spectrum_masking(mel,
                        max_n_masked_band=8,
                        max_n_masked_frame=5):
        assert len(mel.shape) == 3
        n_channel = mel.shape[1]
        n_frame = mel.shape[2]
        for i, _ in enumerate(mel):
            # fill minimum value of spectrogram
            fill_value = torch.min(mel[i])
            for j in range(np.random.randint(1, 4)):
                n_masked_band = np.random.randint(0, max_n_masked_band)
                n_masked_frame = np.random.randint(0, max_n_masked_frame)
                masked_freq_band_start = np.random.randint(0, n_channel - n_masked_band)
                masked_frame_start = np.random.randint(0, n_frame - n_masked_frame)
                mel[i, masked_freq_band_start: masked_freq_band_start + n_masked_band] = fill_value
                mel[i, :, masked_frame_start: masked_frame_start + n_masked_frame] = fill_value
        return mel


class AAMSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, m=0.2, s=30, easy_margin=False):
        super(AAMSoftmax, self).__init__()
        self.test_normalize = True
        self.m = m
        self.s = s
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = torch.tensor(math.cos(self.m))
        self.sin_m = torch.tensor(math.sin(self.m))

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = torch.tensor(math.cos(math.pi - self.m))
        self.mm = torch.tensor(math.sin(math.pi - self.m) * self.m)

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi.half(), (cosine - self.mm).half())

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine).half()
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1, _ = self.accuracy(output.detach(), label.detach(), topk=(1, 5))
        return loss, prec1

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def get_prototype_emb(self):
        return self.weight.data


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=(1,)),
        )  # equals W and b in the paper
        self.linear2 = nn.Sequential(
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=(1,)),
        )
        # equals V and k in the paper

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width,
                                        (kernel_size,), (stride,),
                                        padding, (dilation,),
                                        bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width, momentum=0.8))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      (kernel_size,), (stride,),
                      padding, (dilation,),
                      bias=bias),
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.8)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, f"{channels} % {s} != 0"
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SqzExRes2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SqueezeExcitation(channels)
        )
        self.skip_layer = Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)
        self.residual_layer = Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.net(x)
        res = (self.residual_layer(h) + x)*math.sqrt(0.5)
        skip = self.skip_layer(h)
        return res, skip


class SpeakerEncoder(nn.Module):
    def __init__(self, n_classes=None, n_mels=80, n_mfcc=40, n_channels=256,
                 emb_dim=128, m=0.2, s=30, mel_input=False):
        super().__init__()
        self.n_mels = n_mels
        self.mel_fn = MelFrontend(use_log10=True)
        self.mel_input = mel_input
        dct_mat = torchaudio.functional.create_dct(n_mfcc, n_mels, "ortho").contiguous()
        self.register_buffer('dct_mat', dct_mat)

        self.mfcc_norm = nn.InstanceNorm1d(n_mfcc)
        self.layer1 = Conv1dReluBn(n_mfcc, n_channels, kernel_size=5, padding=2)
        self.layer2 = SqzExRes2Block(n_channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SqzExRes2Block(n_channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SqzExRes2Block(n_channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        self.layer5 = SqzExRes2Block(n_channels, kernel_size=3, stride=1, padding=5, dilation=5, scale=8)
        cat_channels = n_channels * 4
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=(1,))
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2, momentum=0.25)
        self.linear = nn.Linear(cat_channels * 2, emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim, momentum=0.25)
        if n_classes is not None:
            self.aam_softmax = AAMSoftmax(emb_dim, n_classes, m=m, s=s)
        else:
            self.aam_softmax = None

    def forward(self, x, label=None):
        """ Forward pass """
        with torch.no_grad():
            if not self.mel_input:
                x = self.mel_fn(x)
            mfcc = torch.matmul(x.transpose(1, 2), self.dct_mat).transpose(1, 2)
        mfcc = self.mfcc_norm(mfcc)
        out1 = self.layer1(mfcc)
        out2, skip2 = self.layer2(out1)
        out3, skip3 = self.layer3(out2)
        out4, skip4 = self.layer4(out3)
        out5, skip5 = self.layer5(out4)

        out = torch.cat([skip2, skip3, skip4, skip5], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        if label is not None and self.aam_softmax is not None:
            loss, acc = self.aam_softmax(out, label)
            return {"loss": loss, "acc": acc}
        return out

