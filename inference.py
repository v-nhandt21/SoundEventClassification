import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools, os, time, argparse, json
import torch
import tqdm
from torch.utils.data import DataLoader

from utils import AttrDict, build_env
from meldataset import MelDataset, WavDataset
from models import SoundClassifier, Wav2VecClassifier
from utils import scan_checkpoint, load_checkpoint
from ploting import plot_precision_recall_curve, plot_roc_curve, report
import numpy as np

def inference(a, h, wav=False):
     
     device = "cuda"

     if wav:
          model = Wav2VecClassifier(h).to(device)
     else:
          model = SoundClassifier(h).to(device)

     cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
     state_dict = load_checkpoint(cp_g, device)
     model.load_state_dict(state_dict['classifier'])

     if wav:
          testset = WavDataset(h, fileid='/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/test.txt', train=False)
     else:
          testset = MelDataset(h, fileid='/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/test_.txt', train=False)
     test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
     model.eval()

     Y_pred = []
     Y_target = []

     for batch in tqdm.tqdm(test_loader):

          x, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat = model(x)

          pred = torch.argmax(y_hat, -1).item()
          target = torch.argmax(y, -1).item()

          # y_hat = torch.nn.Softmax(dim=1)(y_hat)

          Y_pred.append(y_hat.cpu().detach().numpy())
          Y_target.append(y.cpu().detach().numpy())

     Y_pred = np.concatenate(Y_pred, axis=0 )
     Y_target = np.concatenate( Y_target, axis=0 )

     print(Y_pred.shape)
     print(Y_target.shape)

     classes = {'Motor vehicle (road)': 0, 'Screaming': 1, 'Explosion': 2, 'Female speech': 3, 'Male speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying, sobbing': 7, 'Siren': 8, 'Gunshot, gunfire': 9}
     plt1 = plot_precision_recall_curve(Y_target, Y_pred, list(classes.keys()), "Outdir/test_prc.png")
     plt2 = plot_roc_curve(Y_target, Y_pred, list(classes.keys()), "Outdir/test_roc.png")

     report(Y_target, Y_pred, list(classes.keys()), "Outdir/test_confusion_matrix.png")
     
     

if __name__ == '__main__':
     print('Initializing Training Process..')

     parser = argparse.ArgumentParser()
     parser.add_argument('--checkpoint_path', '-c', default= 'Outdir/Universal_GT_specaug')
     parser.add_argument('--config', default='config_v1.json')
     parser.add_argument('--wav', default=False, type=bool)

     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)
     build_env(a.config, 'config.json', a.checkpoint_path)

     inference(a, h, wav=a.wav)