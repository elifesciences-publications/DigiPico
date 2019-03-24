#!/bin/bash

#
#   Calculate the per well read count for each sample

#

#Suns command
#$ -cwd
#$ -q batchq

S="$1"
W="$2"
readCounter="$3"

PTH=../Analysis/${S}

# Calculate the per well read count with 1kb steps

${readCounter} ${PTH}/Bam/${W}_${S}.bam \
-c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y \
-w 1000 -q 20 > ${PTH}/DigiTitan/RD.${S}/${S}.${W}.wig

echo ${W}" Done!" > ${PTH}/DigiTitan/RD.${S}/${W}.progress.txt

# If done with last well invoke merge script

if [ ${W} == "P024" ]
then
./DigiTitan.mergewig.sh ${S}
fi