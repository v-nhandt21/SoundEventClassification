import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools, os, time, argparse, json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.utils import AttrDict, build_env
from dataloader.meldataset_triplet import MelTripletDataset
from dataloader.wavdataset import WavDataset
from model.models import SoundClassifier
from utils.utils import scan_checkpoint, load_checkpoint, save_checkpoint
from model.loss import FocalLoss, AutoscaleFocalLoss, CrossEntropyWithLogits, LabelSmoothingLoss
from pytorch_balanced_sampler.sampler import SamplerFactory
from inference import inference
from ILI import plainILI



def get_ili_label(filenames, y, LABELS, ILI=True):
     if not ILI:
          return y
     for idx, file in enumerate(filenames):
          if file in LABELS:
               y[idx] = torch.nn.functional.one_hot( torch.tensor(LABELS[file]), num_classes=10)
     return y

def train(rank, a, h):

     ILI = True
     if hasattr(h, 'ILI'):
          print("Set ILI to ", h.ILI)
          ILI = h.ILI

     LABELS = {}

     LogWanDB = True #a.use_wandb

     if LogWanDB:
          import wandb
          wandb.init(sync_tensorboard=True)

     torch.cuda.manual_seed(h.seed)
     device = torch.device('cuda:{:d}'.format(rank))

     criterion = []
     if "FocalLoss" in h.loss:
          criterion.append(FocalLoss())
     if "AutoscaleFocalLoss" in h.loss:
          criterion.append(AutoscaleFocalLoss())
     if "LabelSmoothingLoss" in h.loss:
          criterion.append(LabelSmoothingLoss())

     print("Number of loss: ", len(criterion))

     model = SoundClassifier(h).to(device)

     if LogWanDB:
          wandb.watch(model)

     if rank == 0:
          os.makedirs(a.checkpoint_path, exist_ok=True)
          print("checkpoints directory : ", a.checkpoint_path)

     if os.path.isdir(a.checkpoint_path):
          cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

     steps = 0
     if cp_g is None:
          state_dict = None
          last_epoch = -1
     else:
          print("Resume model")
          state_dict = load_checkpoint(cp_g, device)
          model.load_state_dict(state_dict['classifier'])
          steps = state_dict['steps'] + 1
          last_epoch = state_dict['epoch']

     optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

     if state_dict is not None:
          optim_g.load_state_dict(state_dict['optim_g'])

     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

     f = open(h.input_wavs_train_file, "r", encoding="utf-8")
     audio_files  = f.read().splitlines()
     f.close()
     trainset = MelTripletDataset(h, audio_files, train=True)

     class_idxs = trainset.get_class_idxs()
     
     if hasattr(h, 'balance_alpha'):
          balance_alpha = h.balance_alpha
     else:
          balance_alpha = 0.8

     batch_sampler = SamplerFactory().get(
     class_idxs=class_idxs,
     batch_size=h.batch_size,
     n_batches=int(len(trainset)/h.batch_size),
     alpha=balance_alpha,
     kind='random'
     )

     train_loader = DataLoader(trainset, num_workers=h.num_workers, batch_sampler=batch_sampler, pin_memory=True)

     if rank == 0:
          sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

     model.train()

     for epoch in range(max(0, last_epoch), a.training_epochs):
          if rank == 0:
               start = time.time()
               print("Epoch: {}".format(epoch+1))

          if ILI:
               trainset = MelTripletDataset(h, h.input_wavs_train_file, train=True)
               train_loader = DataLoader(trainset, num_workers=h.num_workers, batch_sampler=batch_sampler, pin_memory=True)
          
          optim_g.zero_grad()

          for i, batch in enumerate(train_loader):
               if rank == 0:
                    start_b = time.time()
          
               x, mel_pos, mel_neg, y, filename = batch

               if ILI:
                    y = get_ili_label(filename, y, LABELS)
               
               x = torch.autograd.Variable(x.to(device, non_blocking=True))
               y = torch.autograd.Variable(y.to(device, non_blocking=True))
               x_pos = torch.autograd.Variable(mel_pos.to(device, non_blocking=True))
               x_neg = torch.autograd.Variable(mel_neg.to(device, non_blocking=True))

               y_hat, emb_anchor = model(x)
               y_hat_pos, emb_pos = model(x_pos)
               y_hat_neg, emb_neg = model(x_neg)

               loss_pos = torch.nn.functional.cosine_similarity(emb_anchor, emb_pos)
               loss_neg = torch.nn.functional.cosine_similarity(emb_anchor, emb_neg)
               loss = torch.relu(loss_pos - loss_neg + 1)

               

               if torch.isnan(y_hat).any():
                    print(filename)
                    # quit()
               else:
                    loss = None
                    for l in criterion:
                         if loss == None:
                              loss = l(y_hat, y) 
                         else:
                              loss += l(y_hat, y)

                    if torch.isnan(loss).any():
                         print(filename)
                         # quit()
                    else:

                         loss.backward()

                         if (i+1) % h.acc_steps == 0:

                              optim_g.step()
                              optim_g.zero_grad()

               if rank == 0:
                    # STDOUT logging
                    if steps % a.stdout_interval == 0:
                         print('Steps : {:d}, Loss: {:4.3f}, s/b : {:4.3f}'.format(steps, loss, time.time() - start_b))

                    # checkpointing
                    if steps % a.checkpoint_interval == 0 and steps != 0:
                         checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                         save_checkpoint(checkpoint_path,
                                        {'classifier': model.state_dict(),
                                        'optim_g': optim_g.state_dict(), 'optim_d': optim_g.state_dict(), 'steps': steps,
                                        'epoch': epoch})
                         model.eval()
                         torch.cuda.empty_cache()
                         inference(a.checkpoint_path, h)
                         if ILI:
                              LABELS = plainILI(a.checkpoint_path, h)
                         model.train()
                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                         sw.add_scalar("loss/train", loss, steps)
                         sw.add_scalar("lr",scheduler_g.get_last_lr()[-1], steps)

                    

               steps += 1

          scheduler_g.step()
          
          if rank == 0:
               print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
     if LogWanDB:
          wandb.finish()

if __name__ == '__main__':
     print('Initializing Training Process..')

     parser = argparse.ArgumentParser()
     parser.add_argument('--checkpoint_path', default= 'Outdir/Universal_GT')
     parser.add_argument('--config', default='config_v1.json')
     parser.add_argument('--training_epochs', default=150, type=int)
     parser.add_argument('--stdout_interval', default=5, type=int)
     parser.add_argument('--checkpoint_interval', default=500, type=int)
     parser.add_argument('--summary_interval', default=10, type=int)
     parser.add_argument('--validation_interval', default=1500, type=int)
     parser.add_argument('--use_wandb', default=False, type=bool)

     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     if os.path.isdir('TMPMeldir'):
          for i in os.listdir("TMPMeldir"):
               os.remove(os.path.join("TMPMeldir", i))
          os.rmdir("TMPMeldir")

     json_config = json.loads(data)
     h = AttrDict(json_config)
     build_env(a.config, 'config.json', a.checkpoint_path)

     torch.manual_seed(h.seed)
     if torch.cuda.is_available():
          torch.cuda.manual_seed(h.seed)
          h.num_gpus = torch.cuda.device_count()
          h.batch_size = int(h.batch_size / h.num_gpus)
          print('Batch size per GPU :', h.batch_size)
     else:
          pass

     train(0, a, h)