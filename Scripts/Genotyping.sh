#!/bin/bash

#
# Genotype known variants on DigiPico data.

#Suns command
#$ -cwd
#$ -q batchq

module add platypus

S="$1"
Hm="$2"
Ht="$3"
Somatic="$4"
Ref="$5"

PTH=../Analysis/${S}

mkdir -p ${PTH}/VC/Genotyping

F="";
for j in ${PTH}/Bam/*_${S}.bam;
do
F=$F,$j
done

echo ${F:1}

platypus callVariants --nCPU=4 --bamFiles=${F:1} \
--getVariantsFromBAMs=0 --source=${Ht} \
--filterReadsWithUnmappedMates 0 --filterReadsWithDistantMates 0 --filterReadPairsWithSmallInserts 0 \
--minPosterior 0 \
--refFile=${Ref} \
--output=${PTH}/VC/Genotyping/${S}.Ht.vcf.gz

platypus callVariants --nCPU=4 --bamFiles=${F:1} \
--getVariantsFromBAMs=0 --source=${Hm} \
--filterReadsWithUnmappedMates 0 --filterReadsWithDistantMates 0 --filterReadPairsWithSmallInserts 0 \
--minPosterior 0 \
--refFile=${Ref} \
--output=${PTH}/VC/Genotyping/${S}.Hm.vcf.gz

platypus callVariants --nCPU=4 --bamFiles=${F:1} \
--getVariantsFromBAMs=0 --source=${Somatic} \
--filterReadsWithUnmappedMates 0 --filterReadsWithDistantMates 0 --filterReadPairsWithSmallInserts 0 \
--minPosterior 0 \
--refFile=${Ref} \
--output=${PTH}/VC/Genotyping/${S}.Somatic.vcf.gz

for i in Ht Hm Somatic;
do

zcat ${PTH}/VC/Genotyping/${S}.${i}.vcf.gz | grep -v "#" | awk 'length($4)+length($5)<5 && $5!~/,/' | awk -v type="${i}" 'BEGIN{OFS="\t"}{
cov=0; var=0;
split($394,VR,":");
for (i=10;i<394;i++){
split($i,V,":");
if (V[5]>0) cov++;
if (V[6]>0) var++;
}
print $1,$2,$3,$4,$5,$6,$7,"SNP-"type,VR[5],VR[6],cov,var
}' | sort -nk 1,1 -nk 2,2 | awk '$11>5 && $12>1' > ${PTH}/VC/Genotyping/${S}.${i}.tsv

done

