#!/bin/sh

# Merge all DigiPico bam files

#Suns command
#$ -cwd
#$ -q batchq

module add samtools
module add picard-tools/2.3.0

S="$1"
L="$2"
Ref="$3"
PTH=../Analysis/${S}

# Check if all bam files have successfully been generated. If not wait until they all are or fix them if they are broken.

cntW=`ls ${PTH}/Bam/????_${S}.bam | wc -l`
cntF=`ls ${PTH}/?0?? | wc -l`

while [ $cntW != 384 ] || [ $cntF != 0 ];
do
 sleep 120
 cntW=`ls ${S}/Bam/????_${S}.bam | wc -l`
 cntF=`ls ${S}/?0?? | wc -l`
 echo "Still Waiting! for: "${S}" > "${cntW}"."${cntF}
  for i in ${PTH}/?0??;
  do
   if (( `stat -c%s ${i}/${S}.Bam` == 92 && `stat -c%s ${i}/${S}.T.Bam` == 175 ));
   then
      echo "Submitting "${S}" "${i:$((${#i}-4))}" "${L}" for redo"
     ./Map.sh ${S} ${i:$((${#i}-4))} ${L} 1 ${Ref}
   fi
  done
done

# Merging DigiPico Bam files

samtools merge ${PTH}/Bam/xMerge_${S}.bam ${PTH}/Bam/????_${S}.bam
samtools index ${PTH}/Bam/xMerge_${S}.bam

