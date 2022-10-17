import os, shutil, torch, glob

class AttrDict(dict):
     def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self


def build_env(config, config_name, path):
     t_path = os.path.join(path, config_name)
     if config != t_path:
          os.makedirs(path, exist_ok=True)
          shutil.copyfile(config, os.path.join(path, config_name))


def load_checkpoint(filepath, device):
     assert os.path.isfile(filepath)
     print("Loading '{}'".format(filepath))
     checkpoint_dict = torch.load(filepath, map_location=device)
     print("Complete.")
     return checkpoint_dict


def save_checkpoint(filepath, obj):
     print("Saving checkpoint to {}".format(filepath))
     torch.save(obj, filepath)
     print("Complete.")


def scan_checkpoint(cp_dir, prefix):
     pattern = os.path.join(cp_dir, prefix + '????????')
     cp_list = glob.glob(pattern)
     if len(cp_list) == 0:
          return None
     return sorted(cp_list)[-1]

def make_weights_for_balanced_classes(images, nclasses):                        
     count = [0] * nclasses
     for item in images:
          for i in range(len(item[1])):
               if item[1][i] == 1:
                    count[i] += 1
          #num += 1
          #if min(count) != 0: break
     weight_per_class = [0.] * nclasses                                      
     N = float(sum(count))                                                   
     for i in range(nclasses):                                                   
          weight_per_class[i] = N/float(count[i])                                 
     weight = [0] * len(images)                                              
     for idx, val in enumerate(images):  
          mx = 0
          for i in range(len(val[1])):
               if val[1][i] == 1:
                    mx = max(mx, weight_per_class[i])
          weight[idx] = mx         
          #if idx > num: break
     return weight