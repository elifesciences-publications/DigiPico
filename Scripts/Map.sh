#!/bin/sh

#
#   Process reads to BAM for DigiPico


#Suns command
#$ -cwd
#$ -q batchq


S="$1"
W="$2"
L="$3"
Redo="$4"
Ref="$5"
len=60
PTH=../Analysis/${S}

module add bowtie2
module add samtools
module add picard-tools/2.3.0
module add trim_galore/0.4.1

mkdir -p ${PTH}/${W}

## Trim reads using Trimgalore
echo "Trim .... $W in $S " 

if [ $L -gt 1 ]
then
  for (( i=1; i<=${L}; i++))
  do	
   file1=${PTH}/Reads/${W}*L00${i}*_R1*.gz
   file2=${PTH}/Reads/${W}*L00${i}*_R2*.gz
   trim_galore --illumina --paired --length $len --clip_R1 12 --clip_R2 12 -o ${PTH}/${W}  ${file1}  ${file2}
  done
else
 file1=${PTH}/Reads/${W}*_R1*.gz
 file2=${PTH}/Reads/${W}*_R2*.gz
 trim_galore --illumina --paired --length $len --clip_R1 12 --clip_R2 12 -o ${PTH}/${W}  ${file1}  ${file2}
fi

## Obtain the path to fastq files

F1="";
F2="";
for j in ${PTH}/${W}/*_R1*.gz;
do
 F1=$F1,$j
done
for j in ${PTH}/${W}/*_R2*.gz;
do
 F2=$F2,$j
done

## Map using Bowtie2
echo "Mapping .."

bowtie2 -p 8 --ignore-quals \
-x ${Ref} \
-1 ${F1:1} -2 ${F2:1} 2> ${PTH}/Reports/${W}_${S}.Report.txt | \
samtools view -bS - | samtools sort -T ${PTH}/${W}/${S}.TMP -o ${PTH}/${W}/${S}.T.Bam

echo "Remove Duplicates .."

MarkDuplicates INPUT=${PTH}/${W}/${S}.T.Bam OUTPUT=${PTH}/Bam/${W}_${S}.bam METRICS_FILE=${PTH}/${W}/${W}.txt TMP_DIR=${PTH}/${W}/
samtools index ${PTH}/Bam/${W}_${S}.bam

## Delete unnecessary files in all has gone well

if (( `stat -c%s ${PTH}/Bam/${W}_${S}.bam.bai` > 50000 )); 
then
 rm -rf ${PTH}/${W}
fi

## Submit the Merge script

if [ $W == "P024" ] && [ $Redo != 1 ];
then
 ./Merge.sh ${S} ${L} ${Ref}
fi