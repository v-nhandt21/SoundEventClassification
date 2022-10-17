import librosa
from transformers import Wav2Vec2Processor
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils import AttrDict
import json, tqdm

class WavDataset(torch.utils.data.Dataset):
     def __init__(self, h, fileid, train=True):
          f = open(fileid, "r", encoding="utf-8")
          self.audio_files = f.read().splitlines()
          f.close()
          random.seed(1234)
          random.shuffle(self.audio_files)
          self.train = train
          self.h = h
          self.transform = None
          self.label_dict = {'Motor vehicle (road)': 0, 'Female speech': 3, 'Male speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying, sobbing': 7, 'Siren': 8, 'Gunshot, gunfire': 9, 'Screaming': 1, 'Explosion': 2}

     def __len__(self):
          return len(self.audio_files)

     def __getitem__(self, idx):
          filename = self.audio_files[idx]

          audio, _ = librosa.load(filename, sr = 16000)
          audio = audio[:int(len(audio)/self.h.hop_size)*self.h.hop_size]
          if audio.shape[0] < self.h.segment_size:
               padding = self.h.segment_size - audio.shape[0]
               offset = padding // 2
               pad_width = (offset, padding - offset)
               audio = np.pad(audio, pad_width, 'constant', constant_values=audio.min())
               # signal = np.pad(signal, pad_width, 'wrap')
          if len(audio) == self.h.segment_size:
               wav_start = 0
          else:
               wav_start = random.randint(0, len(audio) - self.h.segment_size - 1)
          audio = audio[wav_start:wav_start + self.h.segment_size]
          if self.train and self.transform != None:
               audio = self.transform(audio)

          label = filename.split("/")[-1].split("-")[0].split("_")[0]

          label = torch.nn.functional.one_hot( torch.tensor(self.label_dict[label]), num_classes=10)

          return audio, label, filename

     def get_class_idxs(self):
          class_idxs = {}
          for idx in range(len(self.audio_files)):
               filename = self.audio_files[idx]
               label = filename.split("/")[-1].split("-")[0].split("_")[0]
               label = self.label_dict[label]

               if label in class_idxs:
                    class_idxs[label].append(idx)
               else:
                    class_idxs[label] = [idx]
          idxs_list = []
          for k in sorted(class_idxs):
               idxs_list.append(class_idxs[k])
          return idxs_list

if __name__ == "__main__":
     
     with open("/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/Eff_ex4/config_ex4.json") as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)

     #############################
     ds = WavDataset(h, "/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/train.txt", train=False)
     dl = DataLoader(ds, batch_size = 8)
     processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
     for audio, label, filename in dl:
          print(len(audio[0]), len(audio[1]))
          inputs = processor(audio, sampling_rate=16000, return_tensors = "pt", padding = True).input_values
          print(inputs.shape)
          # break
