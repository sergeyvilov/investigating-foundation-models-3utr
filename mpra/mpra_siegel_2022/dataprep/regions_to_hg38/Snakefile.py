import numpy as np
import os
import pandas as pd

#liftover gene coordinates from GRCh37 to GRCh38

regions_tsv = '/s/project/mll/sergey/effect_prediction/MLM/siegel_2022/Beas2B.tsv'

progress_dir = '/s/project/mll/sergey/effect_prediction/MLM/siegel_2022/regions_hg38/' #output dir

utr3_bed = '/s/project/mll/sergey/effect_prediction/MLM/UTR_coords/GRCh38_3_prime_UTR_clean.bed'

liftover_dir = '/s/project/mll/sergey/effect_prediction/tools/liftOver/'

rule all:
    input:
        progress_dir + 'regions_3UTR_GRCh38.bed',


rule get_tss_bed:
    input:
        csv = regions_tsv
    output:
        bed = temp(progress_dir + 'regions_GRCh37.bed')
    shell:
        r'''
        cat {input.csv} |tail -n +2|cut -f3,6|sed -E 's/\t[^\|]*\|/\t/' \
        |awk 'BEGIN{{FS="\t|:|-";OFS="\t"}}{{print $2,$3,$4,$1}}' |sort -V -k1,1 -k2,2|uniq \
        |sed -e 's/^/chr/' -e 's/chrMT/chrM/' > {output.bed}
        '''

rule liftover:
    input:
        bed = progress_dir + 'regions_GRCh37.bed',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'regions_GRCh38.liftover.bed',
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
         bedtools intersect -a {input.regions_bed} -b {input.utr3_bed} -wa -wb|cut -f2,3,6,7,10,4 > {output.bed}

        '''
        

        