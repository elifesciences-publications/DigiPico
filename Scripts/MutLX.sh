#!/bin/bash

#
# Train and variant filtering using MutLX.

#Suns command
#$ -cwd
#$ -q batchq

module add python3

S="$1"

E=20
train_ratio=0.2
PTH=../Analysis/${S}

mkdir -p ${PTH}/MutLX/Q/train/data
mkdir -p ${PTH}/MutLX/Q/train/log
mkdir -p ${PTH}/MutLX/R/train/data
mkdir -p ${PTH}/MutLX/R/train/log
mkdir -p ${PTH}/MutLX/Q/predictions
mkdir -p ${PTH}/MutLX/R/predictions
mkdir -p ${PTH}/MutLX/QF

# tag germline and somatic variants and filter the liw quality varinats.

awk '{M=$1":"$2"."$4">"$5; if (NR==FNR) {a[M]; next} if (M in a) $11="SNP-Hm";
print $0;}' ${PTH}/VC/Genotyping/${S}.Hm.tsv ${PTH}/VC/${S}.tag.tsv | awk '{M=$1":"$2"."$4">"$5; if (NR==FNR) {a[M]; next} if (M in a) $11="SNP-Ht";
print $0;}' ${PTH}/VC/Genotyping/${S}.Ht.tsv - | awk '{M=$1":"$2"."$4">"$5; if (NR==FNR) {a[M]; next} if (M in a) $11="SNP-Somatic";
print $0;}' ${PTH}/VC/Genotyping/${S}.Somatic.tsv - | awk '{
 split($8,P,";");
 split(P[2],FR,"=");
 split(P[3],HP,"=");
 split(P[11],QD,"=");
 split(P[13],Sb,"=");
 split(P[23],R,":");
if($6>60 && FR[2]>0.1 && HP[2]<=4 && QD[2]>10 && Sb[2]<=0.95 && $10/$9>0.1 && R[2]/(R[1]+0.1)>0.1) print $0;
}' - > ${PTH}/MutLX/QF/${S}.QF.tsv 

in=${PTH}/MutLX/QF/${S}.QF.tsv;

#Calculate number of UTDs, Germlines and Somatics+Hets

UC=`grep UTD ${in} | wc -l`
GC=`grep -E "Hm|Ht" ${in} | wc -l`


#Prepare the train file for R-training which will have the same number of UTDs to Germlines (2:1 ratio for Ht:Hm) as well as all the somatics

awk -v u="${UC}" -v g="${GC}" -v s="${S}" 'BEGIN{OFS="\t"; if (u/g<1) {Ratio=u/g; low="U";} else {Ratio=g/u; low="G";}}{
split($8,F,";");
split(F[8],nF,"=");
split(F[9],nR,"=");
split(F[16],TCF,"=");
split(F[17],TCR,"=");
split(F[23],VR,":");
if ($0~/UTD/ && (low=="U" || rand()>(1-Ratio))) {
print $1":"$2"."$4">"$5,0, nF[2], nR[2], TCF[2], TCR[2], VR[1], VR[2], F[22];
}
else {
  RatioAd=Ratio/3;
  if ($0~/Somatic/ || low=="G" || ($0~/Ht/ && rand()>(1-2*RatioAd)) || ($0~/Hm/ && rand()>(1-RatioAd))) {
  print $1":"$2"."$4">"$5,1, nF[2], nR[2], TCF[2], TCR[2], VR[1], VR[2], F[22];
  }
}
}' <(grep -E "SNP|UTD" ${in}) | tr "\\t" "," > ${PTH}/MutLX/R/train/data/${S}.train.csv


#prepare the input file for the R-analysis

awk 'BEGIN{OFS="\t";}{
split($8,F,";");
split(F[8],nF,"=");
split(F[9],nR,"=");
split(F[16],TCF,"=");
split(F[17],TCR,"=");
split(F[23],VR,":");

if ($0~/UTD/) {print $1":"$2"."$4">"$5","0","nF[2]","nR[2]","TCF[2]","TCR[2]","VR[1]","VR[2]","F[22]}
else {print $1":"$2"."$4">"$5","1","nF[2]","nR[2]","TCF[2]","TCR[2]","VR[1]","VR[2]","F[22]}

}' ${in} > ${PTH}/MutLX/R/predictions/${S}.data.csv

#Prepare the train file for Q-training which will have the same number of UTDs to Germlines (2:1 ratio for Ht:Hm) as well as all the somatics

awk -v u="${UC}" -v g="${GC}" -v s="${S}" 'BEGIN{OFS="\t";if (u/g<1) {Ratio=u/g; low="U";} else {Ratio=g/u; low="G";}}{
split($8,F,";");
split(F[1],BRF,"=");split(F[2],FR,"=");
split(F[3],HP,"=");split(F[4],HapSc,"=");
split(F[5],MGOF,"=");split(F[6],MMLQ,"=");
split(F[7],MQ,"=");split(F[11],QD,"=");
split(F[13],Sb,"=");split(F[21],N,":");
Seq=N[1]","N[2]","N[3]","N[4]
asort(N);

if ($0~/UTD/ && (low=="U" || rand()>(1-Ratio))) {
print $1":"$2"."$4">"$5,0,$6,BRF[2],FR[2],HP[2],HapSc[2],MGOF[2],MMLQ[2],MQ[2],QD[2],Sb[2],(N[4]+N[3]+N[2])/21,(N[4]+N[3])/21,N[4]/21,Seq;
}
else {
  RatioAd=Ratio/3;
  if ($0~/Somatic/ || low=="G" || ($0~/Ht/ && rand()>(1-2*RatioAd)) || ($0~/Hm/ && rand()>(1-RatioAd))) {
  print $1":"$2"."$4">"$5,1,$6,BRF[2],FR[2],HP[2],HapSc[2],MGOF[2],MMLQ[2],MQ[2],QD[2],Sb[2],(N[4]+N[3]+N[2])/21,(N[4]+N[3])/21,N[4]/21,Seq;
  }
}

delete N;                     
}' <(grep -E "SNP|UTD" ${in}) | tr "\\t" "," > ${PTH}/MutLX/Q/train/data/${S}.train.csv

#prepare the input file for the Q-analysis

awk -v u="${UC}" -v g="${GC}" -v s="${S}" 'BEGIN{OFS="\t";if (u/g<1) {Ratio=u/g; low="U";} else {Ratio=g/u; low="G";}}{
split($8,F,";");
split(F[1],BRF,"=");split(F[2],FR,"=");
split(F[3],HP,"=");split(F[4],HapSc,"=");
split(F[5],MGOF,"=");split(F[6],MMLQ,"=");
split(F[7],MQ,"=");split(F[11],QD,"=");
split(F[13],Sb,"=");split(F[21],N,":");
Seq=N[1]","N[2]","N[3]","N[4]
asort(N);

if ($11~/UTD/) {
print $1":"$2"."$4">"$5","0","$6","BRF[2]","FR[2]","HP[2]","HapSc[2]","MGOF[2]","MMLQ[2]","MQ[2]","QD[2]","Sb[2]","(N[4]+N[3]+N[2])/21","(N[4]+N[3])/21","N[4]/21","Seq}
else {
print $1":"$2"."$4">"$5","1","$6","BRF[2]","FR[2]","HP[2]","HapSc[2]","MGOF[2]","MMLQ[2]","MQ[2]","QD[2]","Sb[2]","(N[4]+N[3]+N[2])/21","(N[4]+N[3])/21","N[4]/21","Seq}
delete N;                     
}' ${in} > ${PTH}/MutLX/Q/predictions/${S}.data.csv


for mode in Q R;
do

#Perform Training 

echo ${mode}"-Training: "${S}
col=`awk -F'[,]' '{print NF;exit}' ${PTH}/MutLX/${mode}/train/data/${S}.train.csv`
python MutLX/main.py ${S} ${PTH}/MutLX/${mode} ${E} ${train_ratio} ${col} > ${PTH}/MutLX/${mode}/train/log/${S}.train.o 2> ${PTH}/MutLX/${mode}/train/log/${S}.train.e

#Apply the model  

echo ${mode}"-Apply: "${S}"."${i}
pth=${PTH}/MutLX/${mode}/predictions
python MutLX/apply.py ${S} ${PTH}/MutLX/${mode} ${pth} 0.9 ${col} > ${pth}/${S}.apply.o 2> ${pth}/${S}.apply.e

paste ${PTH}/MutLX/QF/${S}.QF.tsv ${pth}/predictions.csv > ${PTH}/MutLX/${S}.MutLX.${mode}.tsv

done

#Extract Optimal Threshold from the log

THQ=`awk '{if ($1~/Optimal_Threshold/) print $2}' ${PTH}/MutLX/Q/predictions/${S}.apply.o`
THR=`awk '{if ($1~/Optimal_Threshold/) print $2}' ${PTH}/MutLX/R/predictions/${S}.apply.o`

echo "Thresholds: "${THQ}" "${THR}

#Filter the data based on the model and calculate stats


paste ${PTH}/MutLX/QF/${S}.QF.tsv <(awk '{print $NF}' ${PTH}/MutLX/${S}.MutLX.Q.tsv) <(awk '{print $NF}' ${PTH}/MutLX/${S}.MutLX.R.tsv) > ${PTH}/MutLX/${S}.MutLX.tsv

awk -v q="${THQ}" -v r="${THR}" '$(NF-1)>q && $NF>r' ${PTH}/MutLX/${S}.MutLX.tsv > ${PTH}/MutLX/${S}.MutLX.Applied.tsv 
