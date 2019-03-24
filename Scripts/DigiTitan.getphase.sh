#!/bin/bash

#
#   Phase Heterozygous Germline variants in DigiPico data
# it first removes all the unphaseables then phases the variants


#Suns command
#$ -cwd
#$ -q batchq

S="$1"
C="$2"

PTH=../Analysis/${S}

mkdir -p ${PTH}/DigiTitan/results/phase
COUNT=${PTH}/DigiTitan/results/titan/tumCounts/tumor_sample
PHASE=${PTH}/DigiTitan/results/phase/${S}.chr${C}.10k.pvcf

echo "Get Phase"

# First Find and remove positions that cant be phased then phase the remaining variants

grep -v "#" ${COUNT}/${S}.${C}.vcf | awk 'BEGIN{OFS="\t"}{
cov=""; var="";
ccov=0; cvar=0; bad=0;
for (M=10;M<394;M++) {split($M,V,":");
  if (V[5]>0) {cov=cov"1"; ccov++;
    if (V[6]/V[5]>0.8) {var=var"1"; cvar++; if (V[6]!=V[5]) bad=1;} else {var=var"0";}
  } else {cov=cov"0"; var=var"0";}
}
if (ccov>4) print $1,$2,$3,$4,$5,cov,var,ccov,cvar,bad;
}' - | awk 'BEGIN{OFS="\t";P[0]=0;P[-1]=0;P[-2]=0;}{
P[NR]=$2;C[NR]=$6;V[NR]=$7;L[NR]=$0;Good=0;
if (NR>3) {
 for (i=1;i<4;i++) {
  if (P[NR-3]<P[NR-3-i]+5000*i) {
   Pen=0; Adv=0;
    for (w=1;w<385;w++) {
     c1=substr(C[NR-3-i],w,1);
     c2=substr(C[NR-3],w,1);
     v1=substr(V[NR-3-i],w,1);
     v2=substr(V[NR-3],w,1);
     Pen=Pen+c1*c2*(v1-v2)*(v2-v1)
     Adv=Adv+c1*c2*v1*v2+c1*c2*(v1-1)*(v2-1)
    }
    if (Adv-Pen>0){
     if (Adv/(Adv-Pen)>=0.8) Good++;
     if (Pen/(Pen-Adv)>=0.8) Good++;
    }
  }
  if (P[NR-i+1]<P[NR-3]+5000*i) {
   Pen=0; Adv=0;
    for (w=1;w<385;w++) {
     c1=substr(C[NR-i+1],w,1);
     c2=substr(C[NR-3],w,1);
     v1=substr(V[NR-i+1],w,1);
     v2=substr(V[NR-3],w,1);
     Pen=Pen+c1*c2*(v1-v2)*(v2-v1)
     Adv=Adv+c1*c2*v1*v2+c1*c2*(v1-1)*(v2-1)
    }
    if (Adv-Pen>0){
     if (Adv/(Adv-Pen)>=0.8) Good++;
     if (Pen/(Pen-Adv)>=0.8) Good++;
    }
  }
 } 
}
if (Good>1) print L[NR-3];
end=NR;
}END{print L[end-2];print L[end-1];print L[end];}' - | awk 'BEGIN{OFS="\t"}{
if (NR==1) {x=1;xx=0;PG=1;PT[NR]=1;P[NR]=$2;C[NR]=$6;V[NR]=$7;print $1,$2,$3,$4,$5,$8,$9,PG"_1",$10;next} 
if (NR<6) {
  if ($2>P[NR-x]+10000) {x=1;PG++;PT[NR]=1;print $1,$2,$3,$4,$5,$8,$9,PG"_1",$10;
  } else {Pen=0; Adv=0;
      for (w=1;w<385;w++) {
       c1=substr(C[NR-x],w,1);
       c2=substr($6,w,1);
       v1=substr(V[NR-x],w,1);
       v2=substr($7,w,1);
       Pen=Pen+c1*c2*(v1-v2)*(v2-v1)
       Adv=Adv+c1*c2*v1*v2+c1*c2*(v1-1)*(v2-1)
      }
      if (Adv+Pen==0) {PT[NR]="";print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;x++;}
      else if (Adv/(Adv-Pen)>=0.8) {PT[NR]=PT[NR-x]; print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;x=1;}
      else if (Pen/(Pen-Adv)>=0.8) {PT[NR]=PT[NR-x]*-1;print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;x=1;}
      else {PT[NR]="";print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;x++;}
  }
  P[NR]=$2;C[NR]=$6;V[NR]=$7;
} 
else {Bad=0; Votes[0]=0; Votes[1]=0; Votes[2]=0; 
  for (Past=1;Past<6;Past++) {
    if ($2<=P[NR-Past-xx]+10000 && PT[NR-Past-xx]!="") {Pen=0; Adv=0; Votes[0]++;
      for (w=1;w<385;w++) {
       c1=substr(C[NR-Past-xx],w,1);
       c2=substr($6,w,1);
       v1=substr(V[NR-Past-xx],w,1);
       v2=substr($7,w,1);
       Pen=Pen+c1*c2*(v1-v2)*(v2-v1)
       Adv=Adv+c1*c2*v1*v2+c1*c2*(v1-1)*(v2-1)
      }
      if (Adv==0 && Pen==0) {Votes[0]=Votes[0]-1;}
      else if (Adv/(Adv-Pen)>=0.8) {PT["V"]=PT[NR-Past-xx]; Votes[length(PT["V"])]++;}
      else if (Pen/(Pen-Adv)>=0.8) {PT["V"]=PT[NR-Past-xx]*-1;Votes[length(PT["V"])]++;}
      else {Votes[0]=Votes[0]-1;}
    } 
    else {Bad++;}
  }
  if (Bad>4 && $2>P[NR-5]+10000) {PG++;PT[NR]=1;print $1,$2,$3,$4,$5,$8,$9,PG"_1",$10;xx=0;} 
  else if (Votes[0]>0) {
    if (Votes[1]/Votes[0]>0.8) {PT[NR]=1;print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;xx=0;} 
    else if (Votes[2]/Votes[0]>0.8) {PT[NR]=-1;print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;xx=0;}
    else {PT[NR]="";print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;xx++;}
  }
  else {PT[NR]="";print $1,$2,$3,$4,$5,$8,$9,PG"_"length(PT[NR]),$10;xx++;}
 P[NR]=$2;C[NR]=$6;V[NR]=$7;
}
}' - | awk '$9==0' > ${PHASE}

echo "De-Noise counts"

# Normalize the reads in the same phase group to have the same WF but retain the existing depth 

awk 'BEGIN{OFS="\t";}{if (NR==FNR) {a[$2]=$8;next}
if ($2 in a) {print $0,a[$2]}
}' ${PHASE} ${COUNT}/tumCounts.chr${C}.txt | grep -v "_0" | awk 'BEGIN{PG=0;split("",R); OFS="\t";}{
split($8,Phase,"_");
if (PG!=Phase[1] || $2-L>100000) {
  if (length(R)>1) {s=0; c=0;
    for (k in R) {
      if (P[k]==1) {s=s+W[k]*R[k]; c=c+W[k];}
      if (P[k]==2) {s=s+W[k]-W[k]*R[k]; c=c+W[k];}
    }
    if (c>0) {avg=s/c} else {avg=0}
    for (k in R) {
      if (P[k]==1) {print D[k],W[k]-int(W[k]*avg),int(W[k]*avg);}
      if (P[k]==2) {print D[k],int(W[k]*avg),W[k]-int(W[k]*avg);}
    }  
  } delete R;
PG=Phase[1]; L=$2;
}
R[$2]=$6/($4+$6); W[$2]=$4+$6; D[$2]=$0; P[$2]=Phase[2];
}' - | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$9,$5,$10,$7}' | \
sort -nk 2,2 | cat <(printf "Chr\tPosition\tRef\tRefCount\tNref\tNrefCount\tNormQuality\n") - > ${COUNT}/tumor_sample.tumCounts.chr${C}.txt

