#!/bin/bash
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
####
cur=$PWD

#############################################################################

for i in $cur/Dataset/silence_data_only/silence_train/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/silence_data_only/silence_val/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/silence_data_only/silence_test/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done
#############################################################################

for i in $cur/Dataset/normalized_dataset_2feb/train_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/normalized_dataset_2feb/val_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

for i in $cur/Dataset/normalized_dataset_2feb/test_labels/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done
#############################################################################

for i in $cur/Dataset/speech_noisy/*.wav;
     do sox -v 0.99 $i -r 16000 -c 1 ${i%.wav}_.wav;
     rm $i;
     mv ${i%.wav}_.wav $i;
done

#################################################################################

# for file in *; do mv "$file" `echo $file | tr ' ' '_'` ; done
# for f in *\(*; do mv -- "$f" "${f//\(/}"; done
# for f in *\)*; do mv -- "$f" "${f//\)/}"; done

# rename 's/speech_noisy/speech-noisy/' *