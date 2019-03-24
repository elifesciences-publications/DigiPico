#!/bin/bash

#
#   Merge all the 1kb read well count data to 100k steps

# 

#Suns command
#$ -cwd
#$ -q batchq

S="$1"

PTH=../Analysis/${S}

IN=${PTH}/DigiTitan/RD.${S}
OUT=${PTH}/DigiTitan/results/readDepth

cnt=`ls ${IN}/*.progress.txt | wc -l`

while [ $cnt != 384 ];
do
sleep 60
echo $cnt
cnt=`ls ${IN}/*.progress.txt | wc -l`
done

awk '{if ($0~/chrom/) {print $0} else {print 0;}}' ${IN}/${S}.A001.wig > ${IN}/Temp.wig

echo "sum all the wells for each bin for "${S} 

for r in A B C D E F G H I J K L M N O P;
do
 for c in 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024;
 do
  echo ${r}${c}
  awk '{if (NR==FNR) {if ($1>0) {a[NR]=1} else {a[NR]=0};next}
  if ($0~/chrom/) {print $0} else {print $1+a[FNR]}
  }' ${IN}/${S}.${r}${c}.wig ${IN}/Temp.wig > ${IN}/Temp2.wig
  rm -rf ${IN}/Temp.wig
  mv ${IN}/Temp2.wig ${IN}/Temp.wig
 done
done

mv ${IN}/Temp.wig ${IN}/${S}.1k.wig

echo "sum every 100 1k to make the 100k well count file"

awk '{
if ($0~/chrom/) {if (NR>1 && P!=1) print S; print $1,$2,$3,"step=100000","span=100000"; S=0; C=0;} 
else {
 if (C==99) {print S+$1; S=0; C=0; P=1;}
 else {S=S+$1; C++; P=0;}
}
}END{print S}' ${IN}/${S}.1k.wig > ${OUT}/tumor_sample.bin100000.wig

rm -rf ${IN}/*.progress.txt
rm -rf ${OUT}/${S}.1k.wig


