#!/bin/sh

# Demultiplexing DigiPico bcl files

#Suns command
#$ -cwd
#$ -q batchq

module add bcl2fastq

ID="$1"
RUN="$2"
SampleSheet="$3"

mkdir -p ../Analysis/${ID}

bcl2fastq -p 24 --runfolder-dir ${RUN}/ --output-dir ../Analysis/${ID}/ --sample-sheet ${SampleSheet}

mv ../Analysis/${ID}/${ID} ../Analysis/${ID}/Reads
