import torch
from speaker_encoder import SpeakerEncoder
import numpy as np
import math
import torchaudio
from scipy.io.wavfile import read

max_wav_value=32768.0

mel = np.load("/data/nhandt23/Workspace/FASTSPEECH/VinTTS-MilVoices_v1.0.0_for_fastspeech/merge_all_20pau/fastspeech2_features_VinTTS-MilVoices_v1.0.0_merge_all_20pau/mel/M-26-N-VinhPhuc0013VinTTS-MilVoices-03366.npy")
log = math.log(10)
mel = mel/log
mel = torch.from_numpy(mel).T.unsqueeze(0)
print(mel.size())

model = SpeakerEncoder(mel_input=True)
checkpoint = torch.load("speaker_encoder.pt", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

emb = model(mel)

np.save("test.npy", emb.cpu().detach().numpy())

api_emb = np.load("/data/nhandt23/Workspace/FASTSPEECH/VinTTS-MilVoices_v1.0.0_for_fastspeech/Speaker_embedding/VC/M-26-N-VinhPhuc0013VinTTS-MilVoices-03366.npy")
api_emb = torch.from_numpy(api_emb)

print(emb)
print(api_emb)

print(torch.eq(emb, api_emb))