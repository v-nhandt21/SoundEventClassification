import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools, os, time, argparse, json
import torch
import tqdm
from torch.utils.data import DataLoader

from utils.utils import AttrDict, build_env
from dataloader.meldataset import MelDataset
from dataloader.wavdataset import WavDataset
from model.models import SoundClassifier, Wav2VecClassifier
from utils.utils import scan_checkpoint, load_checkpoint
from utils.ploting import plot_precision_recall_curve, plot_roc_curve, report
import numpy as np

def plainILI(checkpoint_path, h, wav=False):

     LABELS = {}
     
     device = "cuda"

     if wav:
          model = Wav2VecClassifier(h).to(device)
     else:
          model = SoundClassifier(h).to(device)

     cp_g = scan_checkpoint(checkpoint_path, 'g_')
     state_dict = load_checkpoint(cp_g, device)
     model.load_state_dict(state_dict['classifier'])

     if wav:
          testset = WavDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/train.txt', train=False)
     else:
          testset = MelDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/train.txt', train=False)
     test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
     model.eval()
     for batch in tqdm.tqdm(test_loader):

          x, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat, emb = model(x)

          y_hat = torch.nn.Softmax(dim=1)(y_hat)

          pred = torch.argmax(y_hat, -1).item()
          if pred > 0.9:
               LABELS[filename[0]] = pred 

     return LABELS

def plainILI2(checkpoint_path, h, LABELS={}, train_loader_query=None, wav=False):
     
     print("Lable before update: ", len(LABELS))
     
     device = "cuda"

     if wav:
          model = Wav2VecClassifier(h).to(device)
     else:
          model = SoundClassifier(h).to(device)

     cp_g = scan_checkpoint(checkpoint_path, 'g_')
     state_dict = load_checkpoint(cp_g, device)
     model.load_state_dict(state_dict['classifier'])

     model.eval()
     for batch in tqdm.tqdm(train_loader_query):

          x, mel_pos, mel_neg, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat, emb = model(x)

          y_hat = torch.nn.Softmax(dim=1)(y_hat)

          pred = torch.argmax(y_hat, -1).item()
          target = torch.argmax(y, -1).item()
          if target != pred:
               if pred > 0.85:
                    LABELS[filename[0]] = pred 
               else:
                    if filename[0] in LABELS:
                         LABELS.pop(filename[0])
     print("Update for ", len(LABELS))
     return LABELS


if __name__ == '__main__':
     print('Initializing Training Process..')

     parser = argparse.ArgumentParser()
     parser.add_argument('--checkpoint_path', '-c', default= 'Outdir/Universal_GT_specaug')
     parser.add_argument('--config', default='config_v1.json')
     parser.add_argument('--wav', default=False, type=bool)

     a = parser.parse_args()

     # if os.path.isdir('TMPMeldir'):
     #      for i in os.listdir("TMPMeldir"):
     #           os.remove(os.path.join("TMPMeldir", i))
     #      os.rmdir("TMPMeldir")

     with open(a.config) as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)
     build_env(a.config, 'config.json', a.checkpoint_path)

     inference(a.checkpoint_path, h, wav=a.wav)