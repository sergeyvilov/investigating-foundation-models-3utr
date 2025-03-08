progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/prefiltered/eQTL-susie/' #output dir

utr_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'


rule all:
    input:
        progress_dir + 'eQTL.3utr.bed',

rule get_data:
   '''
   Get Susie credible set for eQTLs
   '''
    output:
        tsv = progress_dir + 'eQTL.tsv.gz',
    log:
        progress_dir + 'download.log'
    shell:
        r'''
        > {output.tsv}
        qts_list=$(wget -q -O - http://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/|grep -Eo 'QTS[0-9]*'|uniq)
        for QTS in $qts_list;do
            echo $QTS >> {log}
            qtd_list=$(wget -q -O - http://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/$QTS|grep -Eo 'QTD[0-9]*'|uniq)
            for QTD in $qtd_list;do
                echo http://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/$QTS/$QTD/$QTD.credible_sets.tsv.gz >> {log}
                wget -q -O - http://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/$QTS/$QTD/$QTD.credible_sets.tsv.gz >> {output.tsv}
            done
        done
        '''

rule get_bed:
    '''
    Filter based on p-value and conver to.bed format
    '''
    input:
        tsv = progress_dir + 'eQTL.tsv.gz',
    output:
        bed = progress_dir + 'eQTL.filtered.bed',
    params:
        p_value_min=1e-12
    shell:
        r'''
        zcat {input.tsv}|grep -v 'gene_id'|awk -v  p_value_min={params.p_value_min} 'BEGIN{{FS="\t";OFS="\t"}}{{if ($8<p_value_min){{print $4"_"$8}}}}'\
        |awk 'BEGIN{{FS="_";OFS="\t"}}{{print $1,$2-1,$2,"GT="$3"/"$4";pvalue="$5}}'> {output.bed}
        '''


rule intersect_3utr:
    '''
    Intersect eQTL bed with 3'UTR regions bed
    '''
    input:
        bed = progress_dir + 'eQTL.filtered.bed',
        utr_bed = utr_bed,
    output:
        bed = progress_dir + 'eQTL.3utr.bed',
    shell:
        r'''
        bedtools intersect -wo -a {input.bed} -b {input.utr_bed} |cut -f1,2,3,4,8|\
        sed -E 's/\tENST/;seq_name=ENST/'|sort -k1,1 > {output.bed}
        '''
