import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools, os, time, argparse, json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.utils import AttrDict, build_env
from dataloader.meldataset import MelDataset
from dataloader.wavdataset import WavDataset
from model.models import SoundClassifier
from utils.utils import scan_checkpoint, load_checkpoint, save_checkpoint
from model.loss import FocalLoss, WeightedFocalLoss, AutoscaleFocalLoss, CrossEntropyWithLogits
from pytorch_balanced_sampler.sampler import SamplerFactory

def train(rank, a, h):

     LogWanDB = True #a.use_wandb

     if LogWanDB:
          import wandb
          wandb.init(sync_tensorboard=True)

     torch.cuda.manual_seed(h.seed)
     device = torch.device('cuda:{:d}'.format(rank))

     if h.loss == "FocalLoss":
          criterion = FocalLoss()
     if h.loss == "AutoscaleFocalLoss":
          criterion = AutoscaleFocalLoss()

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
     trainset = MelDataset(h, h.input_wavs_train_file, train=True)

     class_idxs = trainset.get_class_idxs()

     batch_sampler = SamplerFactory().get(
     class_idxs=class_idxs,
     batch_size=h.batch_size,
     n_batches=int(len(trainset)/h.batch_size),
     alpha=0.8,
     kind='random'
     )

     train_loader = DataLoader(trainset, num_workers=h.num_workers, batch_sampler=batch_sampler, pin_memory=True)

     if rank == 0:
          validset = MelDataset(h, h.input_wavs_val_file, train=False)
          validation_loader = DataLoader(validset, num_workers=1, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
          sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

     model.train()

     for epoch in range(max(0, last_epoch), a.training_epochs):
          if rank == 0:
               start = time.time()
               print("Epoch: {}".format(epoch+1))

          for i, batch in enumerate(train_loader):
               if rank == 0:
                    start_b = time.time()
          
               x, y, filename = batch
               x = torch.autograd.Variable(x.to(device, non_blocking=True))
               y = torch.autograd.Variable(y.to(device, non_blocking=True))

               y_hat, _ = model(x)

               optim_g.zero_grad()

               if torch.isnan(y_hat).any():
                    print(filename)
                    # quit()
               else:
                    loss = criterion(y_hat, y)
                    if torch.isnan(loss).any():
                         print(filename)
                         # quit()
                    else:
                         loss.backward()
                         optim_g.step()

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
                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                         sw.add_scalar("loss/train", loss, steps)
                         sw.add_scalar("lr",scheduler_g.get_last_lr()[-1], steps)

                    # Validation
                    if steps % a.validation_interval == 0:  # and steps != 0:
                         model.eval()
                         torch.cuda.empty_cache()
                         val_err_tot = 0
                         with torch.no_grad():
                              for j, batch in enumerate(validation_loader):
                                   x, y, filename = batch
                                   x = torch.autograd.Variable(x.to(device, non_blocking=True)).float()
                                   y = torch.autograd.Variable(y.to(device, non_blocking=True)).long()

                                   y_hat, _ = model(x)
                                   val_err_tot += criterion(y_hat, y)

                         val_err = val_err_tot / (j+1)
                         sw.add_scalar("loss/val", val_err, steps)

                         model.train()

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
     parser.add_argument('--validation_interval', default=100, type=int)
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