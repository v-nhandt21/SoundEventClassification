import torch
from speaker_encoder import SpeakerEncoder
import numpy as np
import math
import torchaudio
from scipy.io.wavfile import read
import glob
import tqdm
import argparse
import os, sys 
def main():
     # p = argparse.ArgumentParser()
     # p.add_argument('--input_dir', '-i', required=True)
     # p.add_argument('--output_dir', '-o', type=str, default="")
     # args = p.parse_args()
     max_wav_value=32768.0
     mel_input = False
     device = "cuda"

     model = SpeakerEncoder(mel_input=mel_input)
     # checkpoint = torch.load("speaker_encoder.pt", map_location='cpu')
     checkpoint = torch.load("speaker_encoder.pt", map_location=device)
     model.load_state_dict(checkpoint, strict=False)
     model = model.to(device)
     model.eval()

     log = math.log(10)

     output_dir = "/home/noahdrisort/Desktop/DCASE/SoundEventClassification/SpeakerEncoder/test_emb"

     os.makedirs(output_dir, exist_ok=True)

     input_dir = "/home/noahdrisort/Desktop/DCASE/SoundEventClassification/Dataset/test_labels"

     for fi in tqdm.tqdm(glob.glob(input_dir + "/*")):
          try: 
               save_fi = output_dir + "/" + fi.split("/")[-1]
               save_fi = save_fi.replace(".wav", ".npy")

               label = fi.split("/")[-1].split("-")[0]
               # print(label)

               if label != "Female_speech" and label != "Male_speech":
                    continue

               if mel_input:
                    mel = np.load(fi)
                    mel = mel/log
                    mel = torch.from_numpy(mel).T.unsqueeze(0)
                    x = mel.to(device)
               else:
                    wf, sr = torchaudio.load(fi)
                    # print(wf.size())
                    x = wf.to(device)
               emb = model(x)
               emb = emb.cpu().detach().numpy()
               

               np.save(save_fi, emb)
          except:
               continue

if __name__ == "__main__":
     main()

# python gen_emb.py --input_dir /home/nhandt23/Desktop/TTS/TMP/MyTam/wavs_enhanced_22k --output_dir /home/nhandt23/Desktop/TTS/TMP/MyTam/wavs_enhanced_22k_spk_emb