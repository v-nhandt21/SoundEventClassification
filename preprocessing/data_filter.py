import glob, torch
import torchaudio, os

train = glob.glob("Dataset/normalized_dataset_2feb/train_labels/*.wav")
val = glob.glob("Dataset/normalized_dataset_2feb/val_labels/*.wav")
test = glob.glob("Dataset/normalized_dataset_2feb/test_labels/*.wav")

silence_train = glob.glob("Dataset/silence_data_only/silence_train/*.wav")
silence_val = glob.glob("Dataset/silence_data_only/silence_val/*.wav")
silence_test = glob.glob("Dataset/silence_data_only/silence_test/*.wav")

speech_train = glob.glob("Dataset/speech_noisy/*.wav")

print(len(train))
print(len(val))
print(len(test))
# 72354
# 15494
# 15530
os.makedirs("Dataset/script", exist_ok=True)

f_train = open("Dataset/script/train.txt", "w+", encoding="utf-8")
f_val = open("Dataset/script/val.txt", "w+", encoding="utf-8")
f_test = open("Dataset/script/test.txt", "w+", encoding="utf-8")

segment = 65536

segment = 16000 #65536 + 16000

print(segment)

# Dataset/train_labels/others_ogg3200-15007.wav
def check(filename):
     label_dict = {'breaking': 0, \
               'crowd_scream': 1, \
               'crying_sobbing': 2, \
               'explosion': 3, \
               'gunshot_gunfire': 4, \
               'motor_vehicle_road': 5, \
               'siren': 6, \
               'speech': 7, \
               'silence': 8}

     # label = filename.split("/")[-1].split("-")[0].split("_")[0]
     label = filename.split("/")[-1].split("-")[0]
     if label not in label_dict:
          print("Out domain label: ", filename)
          print(label)
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

print("===========")
print(len(test))
for i in train:
     if check(i):
          f_train.write(i + "\n")

for i in speech_train:
     if check(i):
          f_train.write(i + "\n")

for i in silence_train:
     if check(i):
          f_train.write(i + "\n")

for i in val:
     if check(i):
          f_val.write(i + "\n")

for i in silence_val:
     if check(i):
          f_val.write(i + "\n")

for i in test:
     if check(i):
          f_test.write(i + "\n")
     
for i in silence_test:
     if check(i):
          f_test.write(i + "\n")
     
     
     # w, s = torchaudio.load(i)
     # if len(w.size()) != 2:
     #      print(w.size())
     # if w.size(1) >= segment and int(torch.max(w)) <=1:
     #      f_test.write(i + "\n")
     # else:
     #      print(i)

# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (68)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (5)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (65)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (19)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (15)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/siren-siren-10000.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (112)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/crying_sobbing-shah (55)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/crowd_scream-03257.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (91)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (67)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (69)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (64)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (2)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/crowd_scream-02480.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (107)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/gunshot_gunfire-00966.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-00442.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (16)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (66)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/train_labels/explosion-shah (128)_16k.wav
# Short segment:  Dataset/normalized_dataset_2feb/test_labels/explosion-shah (233)_16k.wav
