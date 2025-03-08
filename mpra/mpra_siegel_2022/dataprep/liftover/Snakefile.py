import numpy as np
import os
import pandas as pd

#liftover gene coordinates from GRCh37 to GRCh38

regions_tsv = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/mpra/siegel_2022/siegel_supplemantary/sequence_level_data_Beas2B.csv'

progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/mpra/siegel_2022/preprocessing/regions_hg38/' #output dir

utr3_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean.bed'

liftover_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/tools/liftOver/'

rule all:
    input:
        progress_dir + 'regions_3UTR_GRCh38.bed',
        progress_dir + 'regions_GRCh38.liftover.bed',


rule get_tss_bed:
    input:
        csv = regions_tsv
    output:
        bed = temp(progress_dir + 'regions_GRCh37.bed')
    shell:
        r'''
        cat {input.csv} |tail -n +2 \
        |awk 'BEGIN{{FS=OFS="\""}} {{for (i=1;i<=NF;i+=2) gsub(/,/,"\t",$i)}}1' \
        |cut -f3,6|sed -E 's/(.*)\t.*\|(.*):(.*)-([^"]*).*/chr\2\t\3\t\4\t\1/'   \
        |sort -V -k1,1 -k2,2|uniq \
        |awk 'BEGIN{{OFS="\t"}}{{print $1,$2-1,$3,$4}}' \
        |sed 's/chrMT/chrM/' > {output.bed}
        '''

rule liftover:
    input:
        bed = progress_dir + 'regions_GRCh37.bed',
        chain_file = liftover_dir + 'chain_files/hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = temp(progress_dir + 'regions_GRCh38.liftover.bed'),
        umap = temp(progress_dir + 'regions_GRCh38.umap')
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''

rule intersect_utrs:
    input:
        regions_bed = progress_dir + 'regions_GRCh38.liftover.bed',
        utr3_bed = utr3_bed
    output:
        bed = progress_dir + 'regions_3UTR_GRCh38.bed',
    shell:
        r'''
         bedtools intersect -a {input.regions_bed} -b {input.utr3_bed} -wa -wb|cut -f1,2,3,4,6,7,8,10,13 > {output.bed}

        '''
        

        