#!/bin/bash

#
#   De-novo variant callin using Platypus on DigiPico data


#Suns command
#$ -cwd
#$ -q batchq

module add platypus

S="$1"
C="$2"
Ref="$3"

PTH=../Analysis/${S}

mkdir -p ${PTH}/VC

# Variant calling on DigiPico data using Platypus on specified chromosome

B="";
for j in ${PTH}/Bam/*_${S}.bam;
do
B=$B,$j
done

echo ${B:1}

platypus callVariants --nCPU=4 --bamFiles=${B:1} \
--filterReadsWithUnmappedMates 0 --filterReadsWithDistantMates 0 --filterReadPairsWithSmallInserts 0 \
--countOnlyExactIndelMatches 1 \
--refFile=${Ref} \
--regions=${C} \
--output=${PTH}/VC/${S}_${C}.vcf.gz

# Remove complex variants and long indels, calculate well counts, and extract parametres for MutLX

zcat ${PTH}/VC/${S}_${C}.vcf.gz | grep -v "#" | awk 'length($4)+length($5)<5 && $5!~/,/' | awk 'BEGIN{OFS="\t"}{
N["A"]=0;N["T"]=0;N["C"]=0;N["G"]=0;
split($8,F,";"); split(F[12],SC,"=");
TR=0; TRM=0; TRT=0; TV=0; TVM=0; TVT=0; TVTW=0; Bad=0;
R1=0; R2=0; R3=0; R4=0; R5=0; R6=0; cov=0;
split($394,VR,":");
for (i=10;i<394;i++){
split($i,V,":");
 if (V[5]>0) {cov++;
   if (V[6]==0) { 
   TR++;
    if (V[1]=="0/0") TRM++;
    if (V[1]=="0/1") TRT++;
   }
   if (V[6]>0) {
   TV++;
    if (V[1]=="1/1") TVM++;
    if (V[1]=="0/1") {
    TVT++;
      if (V[5]==V[6]) TVTW++;
    }
    if (V[5]!=V[6]) Bad++;
    if (V[5]==V[6]) R1++;
    if (V[5]==V[6] && V[6]>1) R2++; 
    if (V[5]==V[6] && V[6]>2) R3++; 
    if (V[5]==V[6] && V[6]>3) R4++; 
    if (V[5]==V[6] && V[6]>4) R5++; 
    if (V[5]==V[6] && V[6]>5) R6++; 
   }
 }
}
if (TV>1 && VR[6]>1) {
 for (j in N) {
  for (i=1; i<22; i++) { 
   if (substr(SC[2],i,1)==j) N[j]++;
  }
 }
 print $1,$2,$3,$4,$5,$6,$7,$8";"N["A"]":"N["C"]":"N["G"]":"N["T"]";"TR","TRM","TRT","TV","TVM","TVT","TVTW","Bad","R1","R2","R3","R4","R5","R6";"VR[5]":"VR[6],cov,TV
}
}' > ${PTH}/VC/${S}_${C}.tsv
