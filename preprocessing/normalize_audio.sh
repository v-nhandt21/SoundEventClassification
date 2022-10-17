#!/bin/bash
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
####
cur=$PWD

for i in $cur/Dataset/train_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/val_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/test_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done