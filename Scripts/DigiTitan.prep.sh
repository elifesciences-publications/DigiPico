#!/bin/bash

#
#   Batch Submission of DigiTitan

#Suns command
#$ -cwd
#$ -q batchq

S="$1"
B="$2"
Ref="$3"
readCounter="$4"

PTH=../Analysis/${S}

mkdir -p ${PTH}/DigiTitan/results/readDepth
mkdir -p ${PTH}/DigiTitan/results/titan/hetPosns/tumor_sample
mkdir -p ${PTH}/DigiTitan/results/titan/tumCounts/tumor_sample
mkdir -p ${PTH}/DigiTitan/results/phase
mkdir -p ${PTH}/DigiTitan/RD.${S}

for C in {1..22} X;
do
 echo "TumCounts"
 qsub DigiTitan.hetpos.sh ${S} ${B} ${C} ${Ref}
done

for r in A B C D E F G H I J K L M N O P;
do
 for c in 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024;
 do
  echo "ReadDepth"
  qsub DigiTitan.depth.sh ${S} ${r}${c} ${readCounter}
 done
done

echo "Calculate wig by readCounter for "${B}

${readCounter} ${B} \
-c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y \
-w 100000 -q 20 > ${PTH}/DigiTitan/results/readDepth/normal_sample.bin100000.wig


