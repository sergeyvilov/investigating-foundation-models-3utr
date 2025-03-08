import numpy as np
import os
import pandas as pd

progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/motif_analysis/van_nostrand_2019/eCLIP/' #output dir

liftover_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/tools/liftOver/'

table_3utr = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean.bed'

rule all:
    input:
        progress_dir + 'eCLIP.3utr.pos_IDR.bed', #only IDR
        progress_dir + 'eCLIP.3utr.pos.bed', #positive calls in any replica
        progress_dir + 'eCLIP.3utr.neg.bed',


rule get_tss_bed:
    input:
        meta_tsv = progress_dir + 'ENCSR456FVU_metadata.tsv' #https://www.encodeproject.org/publication-data/ENCSR456FVU/
    output:
        bed = temp(progress_dir + 'eCLIP.hg19.bed6')
    shell:
        r'''
        > {output.bed}
        while read download_url; do
            wget -q $download_url -O - |bgzip -d >> {output.bed}
        done < <(cat {input.meta_tsv} |grep narrowPeak|cut -f21)
        '''

rule liftover:
    input:
        bed = progress_dir + 'eCLIP.hg19.bed6',
        chain_file = liftover_dir + 'chain_files/hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = temp(progress_dir + 'eCLIP.hg38.liftover.bed'),
        umap = temp(progress_dir + 'eCLIP_hg19.umap')
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver -bedPlus=6 {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''


rule extend_50:
    #extend by 50bp in 5' direction and sort
    input:
        bed = progress_dir + 'eCLIP.hg38.liftover.bed',
    output:
        bed = progress_dir + 'eCLIP.hg38.extended.bed',
    shell:
        r'''
        cat {input.bed}|awk 'BEGIN{{OFS="\t"}}{{if ($6=="+") {{$2-=50}} else {{$3+=50}};print}}'|sort -k1,1 -k2,2n > {output.bed}
        '''

# rule split_cell_type:
#     #split IDR  (storng peaks) peaks into cell types
#     input:
#         eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
#     output:
#         eclip_bed = progress_dir + 'eCLIP.hg38.IDR_{cell_type}.bed',
#     shell:
#         r'''
#         grep "_IDR" {input.eclip_bed}|grep  "{wildcards.cell_type}" > {output.eclip_bed}
#         '''

# rule get_positive_merged_set:
#     #positive set: intersection of IDR (storng peaks) with 3'UTR coordinates
#     input:
#         HepG2_bed = progress_dir + 'eCLIP.hg38.IDR_HepG2.bed',
#         K562_bed = progress_dir + 'eCLIP.hg38.IDR_K562.bed',
#         utr_bed =  progress_dir + 'GRCh38.3utr_5Klimited.bed',
#     output:
#         bed = progress_dir + 'eCLIP.3utr.pos_IDR_merged.bed',
#     shell:
#         r'''
#         bedtools intersect -s -a {input.HepG2_bed} -b {input.K562_bed} | bedtools intersect -s -a stdin -b {input.utr_bed} \
#         |sort -k1,1 -k2,2n|bedtools merge -i - | awk 'BEGIN{{OFS="\t"}} {{if ($3-$2>50) {{print}} }}'  \
#         |bedtools intersect -a stdin -b {input.utr_bed} -f 1 -wo \
#         |awk 'BEGIN{{OFS="\t"}} {{print $1,$2,$3,$7,$8,$9}}'|sort -k1,1 -k2,2n|uniq  > {output.bed}
#         '''

rule get_positive_IDR_set:
     #positive set: intersection of IDR (storng peaks) with 3'UTR coordinates
     input:
         eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
         utr_bed =  table_3utr,
     output:
         bed = progress_dir + 'eCLIP.3utr.pos_IDR.bed',
     shell:
         r'''
         grep "_IDR" {input.eclip_bed} | bedtools intersect -s -a stdin -b {input.utr_bed} -wo \
         |awk 'BEGIN{{OFS="\t"}}{{if ($2<$12){{$2=$12}};if ($3>$13) {{$3=$13}}; print $1,$2,$3,$14,$15,$16,$4}}' \
         | sort -k1,1 -k2,2n > {output.bed}
         '''

rule get_positive_set:
    input:
        eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
        utr_bed =  table_3utr,
    output:
        bed = progress_dir + 'eCLIP.3utr.pos.bed',
    shell:
        r'''
        bedtools intersect -s -a  {input.eclip_bed} -b {input.utr_bed} -wo \
        |awk 'BEGIN{{OFS="\t"}}{{if ($2<$12){{$2=$12}};if ($3>$13) {{$3=$13}}; print $1,$2,$3,$14,$15,$16,$19}}' \
        | sort -k1,1 -k2,2n  > {output.bed}
        '''

rule get_negative_set:
    #negative set: all eCLIP (2 replicas+IDR) subtracted from 3'UTR coordinates
    input:
        eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
        utr_bed = table_3utr
    output:
        bed = progress_dir + 'eCLIP.3utr.neg.bed',
    log:
        progress_dir + '/logs/get_negative_set.log'
    shell:
        r'''
        (bedtools subtract -a {input.utr_bed} -b {input.eclip_bed} \
        | cut -f1,2,3,4,5,6,9 |sort -k1,1 -k2,2n  > {output.bed}) > {log} 2>&1
        '''
