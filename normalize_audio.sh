#!/bin/bash
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
####

for i in /home/nhandt23/Desktop/DCASE/Dataset/train_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in /home/nhandt23/Desktop/DCASE/Dataset/val_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in /home/nhandt23/Desktop/DCASE/Dataset/test_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

# FILES=/home/nhandt23/Desktop/DCASE/Dataset/test/*.wav

# for i in $FILES;
#      do sox -v 0.99 "$i" -r 16000 -c 1 ${i%.wav}_.wav;
#      rm "$i";
#      mv ${i%.wav}_.wav "$i";
# done