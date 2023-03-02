import glob, torch
import torchaudio, os

train = glob.glob("Dataset/2feb_additions/train/*.wav")

val = glob.glob("Dataset/2feb_additions/val/*.wav")

test = glob.glob("Dataset/2feb_additions/test/*.wav")

segment = 16000

for filename in train:
     w, s = torchaudio.load(filename)
     if len(w.size()) != 2:
          print("Diff shape: ", filename)
     if w.size(1) < segment:
          print("Short segment: ", filename)

for filename in val:
     w, s = torchaudio.load(filename)
     if len(w.size()) != 2:
          print("Diff shape: ", filename)
     if w.size(1) < segment:
          print("Short segment: ", filename)

for filename in test:
     w, s = torchaudio.load(filename)
     if len(w.size()) != 2:
          print("Diff shape: ", filename)
     if w.size(1) < segment:
          print("Short segment: ", filename)