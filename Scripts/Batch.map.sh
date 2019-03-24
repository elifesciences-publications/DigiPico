#!/bin/bash

#
#   Batch Submission of DigiPico map script

#Suns command
#$ -cwd
#$ -q batchq

S="$1"
Ref="$2"

#Calculates the number of lanes

L=`ls ../Analysis/${S}/Reads/B001*.gz | wc -l`
L=$((L/2))

echo ${S} ${L}

mkdir -p ../Analysis/${S}/Bam

for r in A B C D E F G H I J K L M N O P;
do
 for c in 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024;
 do
   qsub Map.sh ${S} ${r}${c} ${L} 0 ${Ref}
 done
done

