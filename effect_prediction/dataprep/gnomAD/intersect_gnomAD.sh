#!/bin/bash

####################################################
# Intersect gnomAD data with 3'UTR regions         #
# Choose only putative functional (AC=1)           #
# and putative nonfucntional (AC>5%)               #
#                                                  #
#                                                  #
####################################################

gnomAD_light='/lustre/groups/epigenereg01/workspace/projects/vale/tools/gnomAD/v3.1.1_GRCh38/light/GRCh38/gnomAD_GRCh38_light.vcf.gz' #gnomAD with only AF and AC fields in the INFO, works faster

utr3bed=UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed # 3'UTR coordinates

workdir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM/'

cd $workdir


output_dir=gnomAD/

mkdir -p $output_dir

bcftools view -v "snps,indels" $gnomAD_light -R $utr3bed |bgzip -c > ${output_dir}/gnomAD_GRCh38.utr3.vcf.gz

bcftools annotate --threads 4 -h utr3_header.txt -c 'CHROM,FROM,TO,UTR3' -a $utr3bed $gnomAD_light | \
  bcftools query -i '(UTR3!=".")&&((AC=1)||(AF>0.05))' -f "%CHROM\t%POS\tGT=%REF/%ALT;AC=%AC;AF=%AF;seq_name=%UTR3\n"|\
  awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2-1,$2,$3}' > $output_dir/gnomAD_GRCh38.utr3.bed

#bedtools subtract -a ${output_dir}/gnomAD_GRCh38.utr3.bed -b GRASP/GRASP2meQTL.bed -b GRASP/GRASP2eQTL.bed -b clinvar/clinvar.3utr.pathogenic.bed > ${output_dir}/gnomAD.3utr.filtered.bed 
