

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class AutoscaleFocalLoss:
     def __init__(self, threshold=2.0):
          self.threshold = threshold
     
     def gamma(self, logits, threshold = 2):
          return threshold/2 * (torch.cos(np.pi*(logits+1)) + 1)

     def __call__(self, logits, labels):
          assert logits.shape == labels.shape, \
                    "Mismatch in shape, logits.shape: {} - labels.shape: {}".format(logits.shape, labels.shape)
          
          ni = torch.sum(labels, dim = 0)
          pi = ni/torch.sum(ni)
          weight = 1 - pi

          
          logits =  F.softmax(logits, dim=-1)
          CE = - labels * torch.log(logits)
          loss = weight * ((1 - logits)**self.gamma(logits)) * CE
          loss = torch.sum(loss, dim=-1).mean()
          return loss

class LabelSmoothingLoss(torch.nn.Module):
     def __init__(self, smoothing=0.5):

          super(LabelSmoothingLoss, self).__init__()
          self.confidence = 1.0 - smoothing
          self.smoothing = smoothing

     def forward(self, x, target):
          target = torch.argmax(target, -1)
          logprobs = torch.nn.functional.log_softmax(x, dim=-1)
          nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
          nll_loss = nll_loss.squeeze(1)
          smooth_loss = -logprobs.mean(dim=-1)
          loss = self.confidence * nll_loss + self.smoothing * smooth_loss
          return loss.mean()
          
     # def __init__(self, smoothing=0.5):
     #      self.smoothing = smoothing
     #      self.loss = nn.CrossEntropyLoss( label_smoothing= self.smoothing)

     # def __call__(self, logits, labels):
     #      return self.loss(logits, labels)
class FocalLoss:
     def __init__(self, gamma=2.0):
          self.gamma = 2.0

     def __call__(self, logits, labels):
          assert logits.shape == labels.shape, \
                    "Mismatch in shape, logits.shape: {} - labels.shape: {}".format(logits.shape, labels.shape)

          logits =  F.softmax(logits, dim=-1)
          CE = - labels * torch.log(logits)
          loss = ((1 - logits)**self.gamma) * CE
          loss = torch.sum(loss, dim=-1).mean()
          return loss


class PseudoCrossEntropy:
     def __init__(self, temperature):
          self.temperature = temperature
     def __call__(self, logits, labels):
          assert logits.shape == labels.shape, \
                    "Mismatch in shape, logits.shape: {} - labels.shape: {}".format(logits.shape, labels.shape)
          log_softmax_outputs = F.log_softmax(logits/self.temperature, dim=1)
          softmax_targets = F.softmax(labels/self.temperature, dim=1)
          return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class CrossEntropyWithLogits:
     def __init__(self):
          pass
     def __call__(self, logits, labels):
          return F.cross_entropy(logits, labels)

class BinaryCrossEntropyWithLogits:
     def __init__(self):
          pass
     def __call__(self, logits, labels):
          return F.binary_cross_entropy_with_logits(logits, labels)

if __name__ == '__main__':
     # loss = WeightedFocalLoss().to("cuda")
     loss = FocalLoss()
     a = torch.Tensor(50,10).to("cuda")
     b = torch.Tensor(50,10).to("cuda")
     print(loss(a, b))