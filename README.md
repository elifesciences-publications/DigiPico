# DigiPico
Analysis Scripts for DigiPico Sequencing

## INSTRUCTIONS

1) Download the Scripts folder to your working directory
2) Launch all scripts from within the Scripts directory

## Usage

### ===Perparation of DigiPico bam files===

#### Demultiplexing

* The Demultiplexing.sh script generates the DigiPico FastQ files from the bcl files of a paired-end Illumina NGS run.
* You need to specify a sample ID, path to the Illumina run folder, and path to a samplesheet file. 
* The script creates an Analysis directory containing a directory named after the sample ID.
* We recommend a simple ID such as "A111".
* A template samplesheet with all the DigiPico indices is provided in the Scripts directory.
* Please note that you might need to change the "lane" column in the SampleSheet.csv

Requirements:
* bcl2fastq

Usage: 
```
./Demultiplexing.sh [ID] [path/to/run/folder] [path/to/samplesheet.csv]
```
Example:
```
./Demultiplexing.sh A111 ../Runs/180803_ST-E00244_0568_BHN3Y2CCXY SampleSheet.csv
```

#### Mapping using Bowtie2

* The Batch.map.sh script submits 384 jobs to the server to map each of the DigiPico FastQ files to the reference genome.
* If all bam files are generated successfully it will also generate a merged bam file from all of the wells.
* You need to specifiy the sample ID and path to Bowtie2 reference genome in Ensembl format.
* FastQ files must be in the "../Analysis/[ID]/Reads" directory in respect to the Scripts directory.

Requirements:
* Bowtie2
* trim_galore/0.4.1
* picard-tools/2.3.0
* samtools

Usage: 
```
./Batch.map.sh [ID] [path/to/Bowtie2/ref/genome]
```
Example:
```
./Batch.map.sh A111 ../igenomes/GRCh37/Bowtie2Index/genome
```

### ===DigiTitan===

#### Prepare digitised depth and allele count information

* The DigiTitan.prep.sh script submits 23 jobs for the calculation of allele counts at heterozygous positions for each chromosome and 384 jobs for the calculation of digitised depth information. 
* The script also calculates the depth information for the nomral sample from a normal bam file.
* You need to specify the sample ID of the DigiPico run prepared with the preparation scripts, path to a normal Bam file, and a human reference genome fasta file in Ensembl format as well as path to the readCounter executable from HMMCopy utils.

Requirements:
* [Platypus](https://github.com/andyrimmer/Platypus)
* tabix
* [readCounter](https://github.com/shahcompbio/hmmcopy_utils)

Usage: 
```
./DigiTitan.prep.sh [ID] [path/to/normal.bam] [path/to/ref/genome.fa] [path/to/readCounter]
```
Example:
```
./DigiTitan.prep.sh A111 ../normal/Bam/WGS_normal.bam ../igenomes/GRCh37/fasta/genome.fa /bin/hmmcopy_utils/bin/readCounter
```

#### Submit TitanCNA workflow

* The TitanCNA.sh script runs the TitanCNA workflow on the digitised data prepared in the previous step.
* Before running the script you need to install ichorCNA and TitanCNA workflows and edit the essential paths in the "Scripts/TitanCNA/config/config.yaml" file. The essential paths are labelled by an asterik sign at the end of the line.

Requirements:
* [TitanCNA](https://github.com/gavinha/TitanCNA)
* [ichorCNA](https://github.com/broadinstitute/ichorCNA)
* [readCounter](https://github.com/shahcompbio/hmmcopy_utils)

Usage: 
```
./TitanCNA.sh [ID]
```
Example:
```
./TitanCNA.sh A111
```

### ===MutLX===

#### De novo Variant Calling using Platypus

* The Denovo.VC.sh script performs de novo variant calling on DigiPico data using Platypus.
* The script also calculates the well count information as well as other values required in later stages of analysis.
* You need to specify the sample ID of the DigiPico run prepared with the preparation scripts, chromosome number, and a human reference genome fasta file in Ensembl format.
 
Requirements:
* [Platypus](https://github.com/andyrimmer/Platypus)

Usage: 
```
./Denovo.VC.sh [ID] [chromosome] [path/to/ref/genome.fa]
```
Example:
```
for chr in {1..22} X; do ./Denovo.VC.sh A111 ${chr} ../igenomes/GRCh37/fasta/genome.fa; done
```

#### Identify "Unique To DigiPico" Variants

* The Denovo.UTD.sh script performs variant re-calling of DigiPico variant positions on standard WGS data to identify UTD variants.
* The script tags every de novo variant identified in the previous step as either Evd or UTD.
* You need to specify the sample ID of the DigiPico run prepared with the preparation scripts and de novo variant calling script, path to the bam file from standard WGS of normal sample, a comma delimited list of bam files from standard WGS of tumour samples, and a human reference genome fasta file in Ensembl format.
* We recommend using the standard WGS data from the same tumour site as the DigiPico data  
 
Requirements:
* [Platypus](https://github.com/andyrimmer/Platypus)

Usage: 
```
./Denovo.UTD.sh [ID] [path/to/wgs_normal.bam] [path/to/wgs_tumour.bam] [path/to/ref/genome.fa]
```
Example:
```
./Denovo.UTD.sh A111 ../normal/Bam/WGS_normal.bam ../tumour/Bam/WGS_t1.bam,../tumour/Bam/WGS_t2.bam ../igenomes/GRCh37/fasta/genome.fa
```

#### Identify high quality known Somatic and Germline variants in DigiPico data

* The Genotyping.sh scripts perfomrs variant re-calling of known Germline and Somatic variant positions on DigiPico data.
* The list of known homozygous germline, heterzygous germline, and somatic variants must be bgzipped and indexed using tabix.
* The script indentifes and appropriately formats a list of known variants that are supported in DigiPico data for use in MutLX training algorithm. 
* You need to specify the sample ID of the DigiPico run prepared with the preparation scripts and de novo variant calling script, path to the source file for homozygous germline variants, path to the source file for heterozygous germline variants, path to the source file for somatic variants, and a human reference genome fasta file in Ensembl format.
* We recommend using GATK to identify Gemrline variants from normal WGS data, and Strelka2 to idenitify Somatic variants from paired normal-tumour WGS data.

Requirements:
* [Platypus](https://github.com/andyrimmer/Platypus)

Usage: 
```
./Genotyping.sh [ID] [path/to/Homozygous.vcf.gz] [path/to/Heterozygous.vcf.gz] [path/to/Somatic.vcf.gz] [path/to/ref/genome.fa]
```
Example:
```
./Genotyping.sh A111 ../normal/VCF/Hm.vcf.gz ../normal/VCF/Ht.vcf.gz ../tumour/VCF/Somatic.vcf.gz ../igenomes/GRCh37/fasta/genome.fa
```

#### MutLX 

* The MutLX.sh scripts performs Q and R trainign on the previously prepared data and apply the MutLX filters to the de novo variant calls. 
* The algorithm returns a list with all of the de novo variants with respective Q and R probabilties for each variant.
* It also calcualtes and applies optimal Q- and R-thresholds to the data
* You only need to specify the sample ID of the DigiPico run prepared in the previous steps.

Requirements:
* Python3
* Keras

Usage: 
```
./MutLX.sh [ID]
```
Example:
```
./MutLX.sh A111
```
