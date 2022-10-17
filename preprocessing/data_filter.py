import glob, torch
import torchaudio, os

train = glob.glob("Dataset/train_labels/*.wav")

val = glob.glob("Dataset/val_labels/*.wav")

test = glob.glob("Dataset/test_labels/*.wav")

print(len(train))
print(len(val))
print(len(test))
# 72354
# 15494
# 15530
os.makedirs("Outdir", exist_ok=True)

f_train = open("Outdir/train.txt", "w+", encoding="utf-8")
f_val = open("Outdir/val.txt", "w+", encoding="utf-8")
f_test = open("Outdir/test.txt", "w+", encoding="utf-8")

segment = 65536

segment = 16000 #65536 + 16000

print(segment)

# Dataset/train_labels/others_ogg3200-15007.wav
def check(filename):
     label_dict = {'Motor vehicle (road)': 0, 'Female speech': 3, 'Male speech': 4, 'Breaking': 5, 'Crowd': 6, 'Crying, sobbing': 7, 'Siren': 8, 'Gunshot, gunfire': 9, 'Screaming': 1, 'Explosion': 2}
     
     if filename.split("/")[-1].split("-")[0].split("_")[0] not in label_dict:
          return False
     
     w, s = torchaudio.load(filename)

     if len(w.size()) != 2:
          print("Diff shape: ", filename)
          return False

     if w.size(1) < segment:
          print("Short segment: ", filename)
          return False 

     if int(torch.max(w)) > 1:
          print("Un norm: ", filename)
          return False

     return True

for i in train:
     if check(i):
          f_train.write(i + "\n")

for i in val:
     if check(i):
          f_val.write(i + "\n")

for i in test:
     if check(i):
          f_test.write(i + "\n")
     
     
     
     # w, s = torchaudio.load(i)
     # if len(w.size()) != 2:
     #      print(w.size())
     # if w.size(1) >= segment and int(torch.max(w)) <=1:
     #      f_test.write(i + "\n")
     # else:
     #      print(i)
