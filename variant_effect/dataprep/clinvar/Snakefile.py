import pandas as pd

progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/prefiltered/clinvar/' #output dir

clinvar_vcf = '/lustre/groups/epigenereg01/workspace/projects/vale/tools/clinvar/clinvar_20231007.vcf.gz'

utr3_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

rule all:
    input:
        progress_dir + 'clinvar.3utr.pathogenic.bed',

rule filter_calls:
    '''
    Similarly to EVE paper
    use only (Likely) pathogenic/benign variants
    '''
    input:
        vcf = clinvar_vcf,
    output:
        vcf = temp(progress_dir + 'clinvar.filtered.vcf.gz'),
        tbi = temp(progress_dir + 'clinvar.filtered.vcf.gz.tbi'),
    shell:
        r'''
        bcftools view -v "snps,indels"  -e '(CLNSIG!~"benign/i" & CLNSIG!~"pathogenic/i") | CLNSIG~"Conflicting"' {input.vcf} -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule replace_chroms:
    '''
    replace chromosomes according to GRCh38 convention
    '''
    input:
        vcf = progress_dir + 'clinvar.filtered.vcf.gz',
        tbi = progress_dir + 'clinvar.filtered.vcf.gz.tbi',
        chrom_conv = 'chrom_conv.txt',
    output:
        vcf = temp(progress_dir + 'clinvar.new_chroms.vcf.gz'),
        tbi = temp(progress_dir + 'clinvar.new_chroms.vcf.gz.tbi'),
    shell:
        r'''
        bcftools annotate --threads 4 \
        --rename-chrs {input.chrom_conv} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule annotate_regions:
    '''
    add 3'UTR sequence names
    '''
    input:
        vcf = progress_dir + 'clinvar.new_chroms.vcf.gz',
        tbi = progress_dir + 'clinvar.new_chroms.vcf.gz.tbi',
        header = 'headers/utr3_header.txt',
        bed = utr3_bed,
    output:
        vcf = progress_dir + 'clinvar.utr.vcf.gz',
        tbi = progress_dir + 'clinvar.utr.vcf.gz.tbi',
    shell:
        r'''
        bcftools annotate --threads 4 \
        -h {input.header} \
        -c 'CHROM,FROM,TO,=UTR3' \
        -a {input.bed} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule extract_data:
    '''
    select only variants within 3'UTR sequences
    '''
    input:
        vcf = progress_dir + 'clinvar.utr.vcf.gz',
        tbi = progress_dir + 'clinvar.utr.vcf.gz.tbi',
    output:
        tsv = progress_dir + 'clinvar.3utr.pathogenic.bed',
    shell:
        r'''
        bcftools query -i '(UTR3!=".")&&(ALT!=".")' -f "%CHROM\t%POS\tGT=%REF/%ALT;seq_name=%UTR3;CLNREVSTAT=%CLNREVSTAT\t%CLNSIG\n" {input.vcf}|grep 'athogenic'|awk 'BEGIN{{FS="\t";OFS="\t"}}{{print $1,$2-1,$2,$3}}' > {output.tsv}
        '''
