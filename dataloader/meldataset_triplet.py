import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io import wavfile as wf
from pathlib import Path
from torch.utils.data import DataLoader
from utils.utils import AttrDict
import json, tqdm
from aug import get_augment_wav
from dataloader.feature_extractor import get_mel, STFT, get_mel_from_wav
from dataloader.augment import get_transforms

MAX_WAV_VALUE = 32768.0

class MelTripletDataset(torch.utils.data.Dataset):
    def __init__(self, h, audio_files, train=True):

        self.audio_files = audio_files 
        
        random.seed(1234)
        random.shuffle(self.audio_files)
        
        self.train = train
        self.h = h
        self.TMPMeldir = "TMPMeldir"
        
        self.label_dict = {'breaking': 0, \
               'crowd_scream': 1, \
               'crying_sobbing': 2, \
               'explosion': 3, \
               'gunshot_gunfire': 4, \
               'motor_vehicle_road': 5, \
               'siren': 6, \
               'speech': 7, \
               'silence': 8}
               #{'Breaking': 0, 'CrowdOrScream': 1, 'Crying,_sobbing': 2, 'Explosion': 3, 'Gunshot,_gunfire': 4, 'Motor_vehicle_(road)': 5, 'Siren': 6}
                            # { 'Motor_vehicle_(road)': 0, \
                            # 'Screaming': 1, \
                            # 'Explosion': 2, \
                            # 'Female_speech': 3, \
                            # 'Male_speech': 4, \
                            # 'Breaking': 5, \
                            # 'Crowd': 6, \
                            # 'Crying_sobbing': 7, \
                            # 'Siren': 8, \
                            # 'Gunshot_gunfire': 9}
        
        self.stft_obj = STFT(filter_length=h.n_fft, \
                hop_length=h.hop_size, win_length=h.win_size, \
                n_mel_channels=h.num_mels, \
                sampling_rate=h.sampling_rate, \
                mel_fmin=h.fmin, \
                mel_fmax=h.fmax, \
                window='hann')

        if hasattr(h, 'spec_prob'):
            self.transform = get_transforms(
                spec_num_mask=h.spec_num_mask, #3,
                spec_freq_masking=h.spec_freq_masking, #0.15,
                spec_time_masking=h.spec_time_masking, #0.20,
                spec_prob=h.spec_prob
            )
        else:
            self.transform = get_transforms(spec_prob=0)

    def get_label(self, filename):
          return filename.split("/")[-1].split("-")[0]#.split("_")[0]

    def get_data(self, filename):
        _, audio= wf.read(filename)

        audio = get_augment_wav(torch.from_numpy(audio).unsqueeze(0))


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
        
        wav = audio[wav_start:wav_start + self.h.segment_size]
        

        augment_audio = False
        if not augment_audio: # Should not use, can not capture full audio
            file_mel = self.TMPMeldir + "/" + os.path.splitext( os.path.split(filename)[-1])[0] + '.npy'
            if Path(file_mel).is_file() == False:
                os.makedirs(self.TMPMeldir, exist_ok = True)
                file_mel, mel = get_mel_from_wav(wav, filename, self.TMPMeldir, self.h, self.stft_obj)
                np.save(file_mel, mel)
            
            mel = np.load(file_mel)
            mel = torch.from_numpy(mel)
        else:
            file_mel, mel = get_mel_from_wav(wav, filename, self.TMPMeldir, self.h, self.stft_obj)
        
        if len(mel.size()) == 2:
            mel = mel.unsqueeze(0)
            
        if mel.size(2) == 80:
            mel = mel.permute(0,2,1)

        if self.train:
            frames_per_seg = math.ceil(self.h.segment_size / self.h.hop_size)
            mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
            mel = mel[:, :, mel_start:mel_start + frames_per_seg]
            
        label = filename.split("/")[-1].split("-")[0]#.split("_")[0]
        
        if label not in self.label_dict:
            print("Label Unknown: ", label)
            quit()
            self.label_dict[label] = len(self.label_dict)

        label = torch.nn.functional.one_hot( torch.tensor(self.label_dict[label]), num_classes=9)

        if self.train:
            mel = self.transform(mel.squeeze())

        return (mel.squeeze(), label, filename )

    def __getitem__(self, index):

        filename = self.audio_files[index]

        anchor_label = self.get_label(filename)

        filename_pos = self.audio_files[random.randrange(len(self.audio_files))]
        while anchor_label != self.get_label(filename_pos):
          filename_pos = self.audio_files[random.randrange(len(self.audio_files))]
     
        filename_neg = self.audio_files[random.randrange(len(self.audio_files))]
        while anchor_label == self.get_label(filename_neg):
          filename_neg= self.audio_files[random.randrange(len(self.audio_files))]
        # print(filename)
        mel_anchor, label, filename = self.get_data(filename)

        mel_pos, _, _ = self.get_data(filename_pos)

        mel_neg, _, _ = self.get_data(filename_neg)

        return mel_anchor, mel_pos, mel_neg, label, filename
        

    def get_class_idxs(self):
        class_idxs = {}
        for idx in range(len(self.audio_files)):
            filename = self.audio_files[idx]
            label = filename.split("/")[-1].split("-")[0]#.split("_")[0]
            label = self.label_dict[label]

            if label in class_idxs:
                class_idxs[label].append(idx)
            else:
                class_idxs[label] = [idx]
        idxs_list = []
        for k in sorted(class_idxs):
            idxs_list.append(class_idxs[k])
        return idxs_list

    def __len__(self):
        return len(self.audio_files)

# if __name__ == "__main__":
    
#     with open("/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/Eff_ex4/config_ex4.json") as f:
#         data = f.read()
#     json_config = json.loads(data)
#     h = AttrDict(json_config)

#     #############################
#     testset = MelDataset(h, fileid='/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/train.txt', train=False)
#     test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
#     for i, batch in tqdm.tqdm(enumerate(test_loader)):
#         x, y, filename = batch

        
#         x = torch.autograd.Variable(x.to("cuda", non_blocking=True))
        
#         print(x.size())
#         if torch.isnan(x).any():
#             print(filename)
    