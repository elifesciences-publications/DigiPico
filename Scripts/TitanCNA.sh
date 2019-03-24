#!/bin/bash

#
#   DigiTitan 

#Suns command
#$ -cwd
#$ -q batchq



module add python
module add R/3.5.0
module add snakemake

S="$1"

PTH=../Analysis/${S}

mkdir -p ${PTH}/DigiTitan/config

smp=${PTH}/DigiTitan/config/samples.yaml

cp TitanCNA/snakemake/*.snakefile ${PTH}/DigiTitan/
cp TitanCNA/config/config.yaml ${PTH}/DigiTitan/config/config.yaml

echo "samples:" > ${smp}
echo "  tumor_sample:  tumor.bam" >> ${smp}
echo "  normal_sample:  normal.bam" >> ${smp}
echo "" >> ${smp}
echo "" >> ${smp}
echo "pairings:" >> ${smp}
echo "  tumor_sample:  normal_sample" >> ${smp}

(cd ${PTH}/DigiTitan && snakemake -s TitanCNA.snakefile --cores 8)



