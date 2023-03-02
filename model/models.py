from efficientnet_pytorch import EfficientNet
from transformers import Wav2Vec2Model
import torch
import torch.nn as nn

class SoundClassifier(nn.Module):
     
     def __init__(self, h, n_classes=9):     
          super().__init__()  
          self.h = h
          
          self.transform = nn.Linear(1,3)
          self.mv2 = EfficientNet.from_pretrained('efficientnet-b'+str(h.backbone), dropout_rate=0.5)
          
          self.bottleneck = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(1000, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.25),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128))
          self.classifier = nn.Linear(128, n_classes)
     
     def forward(self, x, train=True):

          x = x.unsqueeze(-1)
          x = self.transform(x)
          x = x.permute(0, 3, 1, 2)
          x = self.mv2(x)

          embedding = self.bottleneck(x)
          x = self.classifier(embedding)
          if train:
               x = torch.nn.functional.log_softmax(x, dim=1)
          else:
               x = torch.nn.functional.sigmoid(x)
          return x, embedding

class CRNN(nn.Module):
     
     def __init__(self, h, n_classes=10):     
          super().__init__()  
          self.h = h
          
          self.transform = nn.Linear(1,3)
          self.cnn = EfficientNet.from_pretrained('efficientnet-b'+str(h.backbone), dropout_rate=0.5)
          
          self.bottleneck = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(1000, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.25),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128))
          self.classifier = nn.Linear(128, n_classes)
     
     def forward(self, x):

          x = x.unsqueeze(-1)
          x = self.transform(x)
          x = x.permute(0, 3, 1, 2)
          x = self.cnn(x)

          print(x.size())

          embedding = self.bottleneck(x)
          x = self.classifier(embedding)
          x = torch.nn.functional.log_softmax(x, dim=1)
          return x, embedding
class Wav2VecClassifier(nn.Module):
     
     def __init__(self, h, n_classes=10):     
          super(Wav2VecClassifier, self).__init__()
          self.backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', gradient_checkpointing = False)
          self.backbone.config.mask_time_prob = 0.3
          self.backbone.config.mask_time_min_masks = 5
          self.backbone.config.mask_feature_prob = 0.3
          self.backbone.config.mask_feature_min_masks = 2

          if int(h.freeze_encoder) == 1:
               self.backbone.feature_extractor._freeze_parameters()
          else:
               print("Do not freeze")

          self.bottleneck = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.25),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128))
          self.classifier = nn.Linear(128, n_classes)
     
     def forward(self, x):
          x = self.backbone(x, output_hidden_states = True, return_dict = True)
          x = torch.mean(x.last_hidden_state, dim=1)

          embedding = self.bottleneck(x)
          x = self.classifier(embedding)
          x = torch.nn.functional.log_softmax(x, dim=1)
          return x, embedding

class AttrDict(dict):
     def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

if __name__ == '__main__':

     import json, tqdm
     with open("config_efficient.json") as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)

     ######################

     model = CRNN(h).to("cuda")
     x = torch.randn(8, 32, 80).to("cuda")
     outs = model(x)
     print(outs.size())

     ######################

     # model = Wav2VecClassifier().to("cuda")
     # x = torch.randn(3, 48000).to("cuda")
     # outs = model(x)
     # print(outs.size())