#!/bin/bash

#
#   Find Het sites from blood and call the variants for Titan


#Suns command
#$ -cwd
#$ -q batchq

module add platypus
module add tabix

S="$1"
B="$2"
C="$3"
Ref="$4"

PTH=../Analysis/${S}
HETP=${PTH}/DigiTitan/results/titan/hetPosns/tumor_sample
COUNT=${PTH}/DigiTitan/results/titan/tumCounts/tumor_sample

## Variant calling on normal Bam to find the hetereozygous positions

echo "Variant calling on chr"${C}" of "${B}" for "${S}

platypus callVariants --nCPU=4 --bamFiles=${B} \
--minPosterior 60 \
--refFile=${Ref} \
--regions=${C} \
--output=${HETP}/${C}.T.vcf

# Extract heterozygous positions and generate source file for variant recalling

grep -v "#" ${HETP}/${C}.T.vcf | awk '$5!~/,/ && length($4)<4 && length($5)<4' | awk 'BEGIN{OFS="\t";}{
split($10,P,":");
if (P[5]>9 && P[6]>2 && P[6]/P[5]>0.3 && P[6]/P[5]<0.7) print $1,$2,$3,$4,$5,$6;
}' | bgzip > ${HETP}/${C}.vcf.gz

tabix -p vcf ${HETP}/${C}.vcf.gz
rm -rf ${HETP}/${C}.T.vcf

# Variant re-calling of Het positions on DigiPico data

F="";
for j in ${PTH}/Bam/*_${S}.bam;
do
F=$F,$j
done

echo "Genotyping on "${C}" of "${S}

platypus callVariants --nCPU=4 --bamFiles=${F:1} \
--getVariantsFromBAMs=0 --source=${HETP}/${C}.vcf.gz \
--minPosterior 0 \
--refFile=${Ref} \
--regions=${C} \
--output=${COUNT}/${S}.${C}.vcf

awk '$0!~/#/' ${COUNT}/${S}.${C}.vcf | awk 'BEGIN{OFS="\t";}{
cov=0; var=0; 
for (i=10;i<394;i++) {
 split($i,P,":");
 if (P[5]>0) {cov++;
   if (P[6]/P[5]>0.8) var++;
 }
}
if (cov>4) print $1":"$2"."$4">"$5,cov-var,var;
}' | awk 'BEGIN{OFS="\t";}{
if (NR==FNR) {R[$1]=$2; V[$1]=$3;next}
if ($1":"$2"."$4">"$5 in R) {print $1,$2,$4,R[$1":"$2"."$4">"$5],$5,V[$1":"$2"."$4">"$5],$6}
}' - <(zcat ${HETP}/${C}.vcf.gz) > ${COUNT}/tumCounts.chr${C}.txt

./DigiTitan.getphase.sh ${S} ${C} 
