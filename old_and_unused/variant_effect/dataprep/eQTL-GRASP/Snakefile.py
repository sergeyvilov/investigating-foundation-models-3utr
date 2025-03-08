progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/prefiltered/GRASP/' #output dir

dbSNP = '/lustre/groups/epigenereg01/workspace/projects/vale/tools/dbSNP/GRCh38/00-common_all.vcf.gz'

utr_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

rule all:
    input:
        expand(progress_dir + 'GRASP2{QTLtype}.3utr.bed',QTLtype=['eQTL','meQTL']),

wildcard_constraints:
    QTLtype="eQTL|meQTL"

rule get_data:
    output:
        tsv = progress_dir + 'GRASP2{QTLtype}s',
    shell:
        r'''
        wget -q https://grasp.nhlbi.nih.gov/downloads/GRASP2_{wildcards.QTLtype}_results.zip -O {output}.zip
        unzip -d {progress_dir} -o {output}.zip
        rm {output}.zip
        '''

rule get_dbSNP_id:
    input:
        tsv = progress_dir + 'GRASP2{QTLtype}s',
    output:
        tsv_ids = progress_dir + 'GRASP2{QTLtype}_ids.tsv',
        tsv_pvalues = progress_dir + 'GRASP2{QTLtype}_pvalues.tsv',
    #params:
        #p_value_min=1e-12
    shell:
        r'''
        #cat {input.tsv}|tail -n +2|\
        #awk -v  p_value_min=NONE 'BEGIN{{FS="\t"}}{{if ($11<p_value_min){{print $9}}}}' > {output.tsv_ids}
        cat {input.tsv}|tail -n +2|cut -f9 > {output.tsv_ids} 
        cat {input.tsv}|tail -n +2|cut -f9,11|sed 's/E/e/'|sort -g -k2,2 > {output.tsv_pvalues} 
        '''

rule intersect_dbSNP:
    input:
        tsv_ids = progress_dir + 'GRASP2{QTLtype}_ids.tsv',
        tsv_pvalues = progress_dir + 'GRASP2{QTLtype}_pvalues.tsv',
        dbSNP = dbSNP
    output:
        bed = progress_dir + 'GRASP2{QTLtype}.dbSNP.bed',
    shell:
        r'''
        bcftools view -i "ID=@{input.tsv_ids}" -v  "snps,indels" --max-alleles 2 {input.dbSNP} \
        |grep -v '#'|awk 'BEGIN{{FS="\t";OFS="\t"}}{{print "chr"$1,$2-1,$2,"GT="$4"/"$5,$3}}' > {output.bed}
        '''

rule add_ids:
    input:
        bed = progress_dir + 'GRASP2{QTLtype}.dbSNP.bed',
        tsv_pvalues = progress_dir + 'GRASP2{QTLtype}_pvalues.tsv',
    output:
        bed = progress_dir + 'GRASP2{QTLtype}.hg38.bed',
    shell:
        r'''
        awk 'BEGIN{{OFS="\t"}} NR==FNR {{h[$1] = $2; next}} {{print $1,$2,$3,$4";id="$5";pvalue="h[$5]}}' {input.tsv_pvalues} {input.bed} > {output.bed}
        '''

rule intersect_3utr:
    input:
        bed = progress_dir + 'GRASP2{QTLtype}.hg38.bed',
        utr_bed = utr_bed,
    output:
        bed = progress_dir + 'GRASP2{QTLtype}.3utr.bed',
    shell:
        r'''
        bedtools intersect -wo -a {input.bed} -b {input.utr_bed} |cut -f1,2,3,4,8|\
        sed -E 's/\tENST/;seq_name=ENST/'|sort -k1,1 > {output.bed}
        '''
