import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io import wavfile as wf 
from preprocess import get_mel, STFT, get_mel_from_wav
from pathlib import Path
import glob
from augment import get_transforms

MAX_WAV_VALUE = 32768.0

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, h, fileid, train=True):

        f = open(fileid, "r", encoding="utf-8")
        self.audio_files = f.read().splitlines()
        f.close()
        
        random.seed(1234)
        random.shuffle(self.audio_files)
        
        self.train = train
        self.h = h
        self.TMPMeldir = "TMPMeldir"
        
        self.label_dict = {'Motor vehicle (road)': 0, 'Screaming': 1, 'Explosion': 2, 'Female speech': 3, 'Male speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying, sobbing': 7, 'Siren': 8, 'Gunshot, gunfire': 9}
        
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

    def __getitem__(self, index):

        filename = self.audio_files[index]
        _, audio= wf.read(filename)


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

        audio = wav / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        if len(audio.size()) == 3:
            audio = audio[:,:,0]
        
        file_mel = self.TMPMeldir + "/" + os.path.splitext( os.path.split(filename)[-1])[0] + '.npy'
        if Path(file_mel).is_file() == False:
            os.makedirs(self.TMPMeldir, exist_ok = True)
            file_mel = get_mel_from_wav(wav, filename, self.TMPMeldir, self.h, self.stft_obj)
        
        mel = np.load(file_mel)
        mel = torch.from_numpy(mel)
        
        if len(mel.size()) == 2:
            mel = mel.unsqueeze(0)
            
        if mel.size(2) == 80:
            mel = mel.permute(0,2,1)

        if self.train:
            frames_per_seg = math.ceil(self.h.segment_size / self.h.hop_size)
            assert audio.size(1) >= self.h.segment_size
            mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
            mel = mel[:, :, mel_start:mel_start + frames_per_seg]
            
        label = filename.split("/")[-1].split("-")[0].split("_")[0]
        
        if label not in self.label_dict:
            print("Label Unknown")
            quit()
            self.label_dict[label] = len(self.label_dict)

        label = torch.nn.functional.one_hot( torch.tensor(self.label_dict[label]), num_classes=10)

        if self.train:
            mel = self.transform(mel.squeeze())

        return (mel.squeeze(), label, filename )

    def __len__(self):
        return len(self.audio_files)

import librosa
from transformers import Wav2Vec2Processor

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

if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    from utils import AttrDict
    import json, tqdm
    with open("/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/Eff_ex4/config_ex4.json") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    #############################
    # ds = WavDataset(h, "/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/train.txt", train=False)
    # dl = DataLoader(ds, batch_size = 8)
    # processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    # for audio, label, filename in dl:
    #     print(len(audio[0]), len(audio[1]))
    #     inputs = processor(audio, sampling_rate=16000, return_tensors = "pt", padding = True).input_values
    #     print(inputs.shape)
    #     # break

    #############################
    testset = MelDataset(h, fileid='/home/nhandt23/Desktop/DCASE/SoundClasification/Outdir/train.txt', train=False)
    test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True)
    for i, batch in tqdm.tqdm(enumerate(test_loader)):
        x, y, filename = batch

        
        x = torch.autograd.Variable(x.to("cuda", non_blocking=True))
        
        print(x.size())
        if torch.isnan(x).any():
            print(filename)
    