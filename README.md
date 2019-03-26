# DigiPico
Analysis Scripts for DigiPico Sequencing Data

## MutLX
MutLX tool for accrate identification of true islet specific variants from DigiPico data. 

### Requirements


### Instalation


### Input

A csv file in the following format without header:

Mutation,Type,[41 MutLX features extracted from DigiPico data]

Type column must be of any of the following categories:
* SNP-Unq : For UTD variants
* SNP-Hm : For homozygous germline variants
* SNP-Ht-H : For high-confidence heterezygous germline variants (Important for tumours with complex genomes)
* SNP-Ht : For other heterozygous germline variants
* SNP-Somatic : For known somatic mutations

Example files for analysis with DigiPico can be downloaded from below links:
* [Sample file for DigiPico run without expected islet specific variants](https://drive.google.com/open?id=11m_fSPoW2oqmk8H8Ffqpu2FdN9dfuyet)
* [Sample file for DigiPico run with expected islet specific variants](https://drive.google.com/open?id=1j2LFKdEDBOrWKA2yG525jWQlfDOlTnvb)

### Running MutLX

```
python mutLX.py --input test1.csv --out_path test1_Results --sample_name DigiPico_test1
```

### Arguments

* --input : Path to input csv file
* --out_path : Outpu path (default = run directory)
* --sample_name : Sample name to be used as prefix for output files (default = "DigiPico")
* --batch_size : Training batch size (default = 8) 
* --epochs : Number of epochs in training (default = 10)
* --subset_num : Number of training subsets to be used for training (default = 25)
* --drop_it : Number of iterations for dropout analysis (default = 100)

### Output
