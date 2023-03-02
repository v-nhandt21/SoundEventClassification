
f = open("/home/noahdrisort/Desktop/DCASE/ECAPA-TDNN/19jan_sre/train.txt", "r", encoding="utf-8")
lines = f.read().splitlines()
fw= open("train_new.txt", "w+", encoding="utf-8")
with open("train.txt", "r", encoding="utf-8") as f2:
     lili = f2.read().splitlines()
     for line in lili:
          if line in lines:
               fw.write(line+"\n")
