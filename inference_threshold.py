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
from sklearn.metrics import precision_recall_curve
import numpy as np

def inference(checkpoint_path, h, wav=False):
     
     device = "cuda"
     classes = {'Motor_vehicle_(road)': 0, 'Screaming': 1, 'Explosion': 2, 'Female_speech': 3, 'Male_speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying_sobbing': 7, 'Siren': 8, 'Gunshot_gunfire': 9}

     if wav:
          model = Wav2VecClassifier(h).to(device)
     else:
          model = SoundClassifier(h).to(device)

     cp_g = scan_checkpoint(checkpoint_path, 'g_')
     state_dict = load_checkpoint(cp_g, device)
     model.load_state_dict(state_dict['classifier'])

     if wav:
          testset = WavDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/test.txt', train=False)
          valset = WavDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/val.txt', train=False)
     else:
          testset = MelDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/test.txt', train=False)
          valset = MelDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/EfficientNet/ex13/val.txt', train=False)

     test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
     val_loader = DataLoader(valset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
     model.eval()

     Y_pred, Y_target = [], []

     f_meta = open(checkpoint_path + "/meta.tsv", "w+", encoding="utf-8")
     f_embedding = open(checkpoint_path + "/embedding.tsv", "w+", encoding="utf-8")

     ############################################### Validation -> Find Threshold
     Y_pred_val, Y_target_val = [], []
     for batch in tqdm.tqdm(val_loader):
          x, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat, emb = model(x, train=False)
          # pred = torch.argmax(y_hat, -1).item() 
          # target = torch.argmax(y, -1).item()
          y_hat = torch.nn.Softmax(dim=1)(y_hat)

          Y_pred_val.append(y_hat.cpu().detach().numpy())
          Y_target_val.append(y.cpu().detach().numpy())

     Y_pred_val = np.concatenate(Y_pred_val, axis=0 )
     Y_target_val = np.concatenate( Y_target_val, axis=0 )

     print(Y_pred_val.shape)
     print(Y_target_val.shape)

     BestThreshold = []

     for i in range(len(classes)):
          precision, recall, thresholds = precision_recall_curve(
               np.argmax(Y_target_val, axis=1), Y_pred_val[:, i], pos_label=i)
          f1 = 2 * precision * recall / (precision + recall)
          best_idx = np.argmax(f1)
          BestThreshold.append(thresholds[best_idx])
     print(BestThreshold)

     ############################################### Prediction
     for batch in tqdm.tqdm(test_loader):

          x, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat, emb = model(x, train=False)

          pred = torch.argmax(y_hat, -1).item()
          target = torch.argmax(y, -1).item()

          y_hat = torch.nn.Softmax(dim=1)(y_hat)

          ########### Prediction Control
          priority = [2, 1, 8, 7, 0, 6, 9, 5, 3, 4]
          for p in priority:
               if y_hat[:,p] > BestThreshold[p]:
                    y_hat = torch.nn.functional.one_hot( torch.tensor(p), num_classes=10).unsqueeze(0)
          
          f_meta.write(str(target) + "\n")
          emb = emb[0]
          f_embedding.write("\t".join([str(s) for s in emb.tolist()]) + "\n")

          Y_pred.append(y_hat.cpu().detach().numpy())
          Y_target.append(y.cpu().detach().numpy())

     Y_pred = np.concatenate(Y_pred, axis=0 )
     Y_target = np.concatenate( Y_target, axis=0 )

     print(Y_pred.shape)
     print(Y_target.shape)

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

     # if os.path.isdir('TMPMeldir'):
     #      for i in os.listdir("TMPMeldir"):
     #           os.remove(os.path.join("TMPMeldir", i))
     #      os.rmdir("TMPMeldir")

     with open(a.config) as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)
     # build_env(a.config, 'config.json', a.checkpoint_path)

     inference(a.checkpoint_path, h, wav=a.wav)