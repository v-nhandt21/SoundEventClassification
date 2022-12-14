import glob
import numpy as np

f_meta = open("meta.tsv", "w+", encoding="utf-8")
f_embedding = open("embedding.tsv", "w+", encoding="utf-8")

for fi in glob.glob("train_emb/*"):
     emb = np.load(fi)

     label = fi.split("/")[-1].split("-")[0]
     # print(label)

     if label == "Female_speech":
          target = 3
     if label == "Male_speech":
          target = 4

     f_meta.write(str(target) + "\n")
     emb = emb[0]
     f_embedding.write("\t".join([str(s) for s in emb.tolist()]) + "\n")