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

def inference(checkpoint_path, h, wav=False):
     
     device = "cuda"

     if wav:
          model = Wav2VecClassifier(h).to(device)
     else:
          model = SoundClassifier(h).to(device)

     cp_g = scan_checkpoint(checkpoint_path, 'g_')
     state_dict = load_checkpoint(cp_g, device)
     model.load_state_dict(state_dict['classifier'])

     if wav:
          testset = WavDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Dataset/19jan_sre_22/test.txt', train=False)
     else:
          testset = MelDataset(h, fileid='/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Dataset/script/test.txt', train=False)
     test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
     model.eval()

     Y_pred = []
     Y_target = []

     f_meta = open(checkpoint_path + "/meta.tsv", "w+", encoding="utf-8")
     f_embedding = open(checkpoint_path + "/embedding.tsv", "w+", encoding="utf-8")

     fail = 0

     for batch in tqdm.tqdm(test_loader):

          x, y, filename = batch
          x = torch.autograd.Variable(x.to(device, non_blocking=True))
          y_hat, emb = model(x)

          pred = torch.argmax(y_hat, -1).item()
          target = torch.argmax(y, -1).item()

          y_hat = torch.nn.Softmax(dim=1)(y_hat)
          
          f_meta.write(str(target) + "\n")
          emb = emb[0]
          f_embedding.write("\t".join([str(s) for s in emb.tolist()]) + "\n")

          Y_pred.append(y_hat.cpu().detach().numpy())
          Y_target.append(y.cpu().detach().numpy())

          if pred != target :# and target == 5:
               fail+=1
               # print(filename[0])
               fi = str(filename[0])
               os.system("scp " + fi +" /home/noahdrisort/Desktop/DCASE/SoundEventClassification/Outdir/False_Sample")

     
     print(fail)

          # print(y_hat.cpu().detach().numpy().shape)
          # print(y.cpu().detach().numpy().shape)

     Y_pred = np.concatenate(Y_pred, axis=0 )
     Y_target = np.concatenate( Y_target, axis=0 )

     print(Y_pred.shape)
     print(Y_target.shape)

     classes = {'breaking': 0, \
               'crowd_scream': 1, \
               'crying_sobbing': 2, \
               'explosion': 3, \
               'gunshot_gunfire': 4, \
               'motor_vehicle_road': 5, \
               'siren': 6, \
               'speech': 7, \
               'silence': 8}
     # {'Motor_vehicle_(road)': 0, 'Screaming': 1, 'Explosion': 2, 'Female_speech': 3, 'Male_speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying_sobbing': 7, 'Siren': 8, 'Gunshot_gunfire': 9}
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
     build_env(a.config, 'config.json', a.checkpoint_path)

     inference(a.checkpoint_path, h, wav=a.wav)