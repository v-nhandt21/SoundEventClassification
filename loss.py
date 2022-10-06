

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
          
class WeightedFocalLoss(nn.Module):
     "Non weighted version of Focal Loss"
     def __init__(self, alpha=.25, gamma=2):
          super(WeightedFocalLoss, self).__init__()
          self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
          self.gamma = gamma

     def forward(self, inputs, targets):
          BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
          y = torch.ones(targets.shape).cuda()
          targets = torch.where(targets == 0, targets, y)
          
          targets = targets.type(torch.long)
          
          at = self.alpha.gather(0, targets.data.view(-1))
          pt = torch.exp(-BCE_loss)

          at = at.view(-1, 56)
          
          F_loss = at*(1-pt)**self.gamma * BCE_loss
          return F_loss