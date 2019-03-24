#!/bin/bash

#
#   Identify the list of UTD variants from Platypus and tag a Denovo file for the variants


#Suns command
#$ -cwd
#$ -q batchq

module add tabix
module add platypus

S="$1"
B="$2"
W="$3"

PTH=../Analysis/${S}

echo ${S}" against: "${B}" and "${W}

# Prepare the source file from DigiPico variants calls

rm -rf ${PTH}/VC/${S}.T.tsv
for i in {1..22} X;
do
 cat ${PTH}/VC/${S}_${i}.tsv >> ${PTH}/VC/${S}.T.tsv
done

mv ${PTH}/VC/${S}.T.tsv ${PTH}/VC/${S}.tsv

bgzip ${PTH}/VC/${S}.tsv
tabix -p vcf ${PTH}/VC/${S}.tsv.gz

# Count the number of Bam files

OIFS=$IFS
IFS=','
Wc=0;
for x in $W
do
 Wc=$((Wc+1))
done
IFS=$OIFS

# Joint variant re-calling of DigiPico variants on normal and WGS bam files

platypus callVariants --nCPU=4 --bamFiles=${W} \
--getVariantsFromBAMs=0 --source=${PTH}/VC/${S}.tsv.gz \
--minPosterior 0 --minMapQual 5 \
--refFile=${Ref} \
--output=${PTH}/VC/${S}.UTD.vcf.gz

zcat ${PTH}/VC/${S}.UTD.vcf.gz | grep -v "#" | awk -v wc="${Wc}" '{
split($10,B,":");
if (B[6]>0) {print $1":"$2"\tEvd-Germline"; next} 
for (i=1;i<wc;i++) {
split($(10+i),T,":");
 if (T[6]>0) {print $1":"$2"\tEvd-Somatic"; next}
}
}' | awk 'BEGIN{OFS="\t"}{if (NR==FNR) {a[$1]=$2;next}
if ($1":"$2 in a) $11=a[$1":"$2]; 
print $0;
}' - <(zcat ${S}/VC/Denovo/${S}.tsv.gz) | awk 'BEGIN{OFS="\t"}{if (NR==FNR) {a[$1":"$2]=$8;next}
if ($1":"$2 in a) $11=a[$1":"$2];
print $0;
}' <(awk '$11>5 && $12>1' ${S}/VC/Genotyping/${S}.*.tsv) - | awk 'BEGIN{OFS="\t"}{
if (NF==10) $11="SNP-Unq";
print $0
}' > ${S}/VC/Denovo/${S}.tag.tsv

awk 'BEGIN{m=0;t=0;s=0;u=0;OFS="\t";}{
if ($0~/SNP-Hm/) m++;
if ($0~/SNP-Ht/) t++;
if ($0~/SNP-Somatic/) s++;
if ($0~/SNP-Unq/) u++;
}END{print m,t,s,u}' ${S}/VC/Denovo/${S}.tag.tsv > ${S}/VC/UTD.Report



