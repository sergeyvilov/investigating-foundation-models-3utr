progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/prefiltered/CADD/' #output dir

utr_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

wildcard_constraints:
    label="pos|neg",
    vartype="snps|indels"

rule all:
    input:
        expand(progress_dir + 'CADD.3utr.{vartype}.{label}.bed',label=['pos','neg'],vartype=['snps','indels']),
        expand(progress_dir + 'CADD.3utr.{vartype}.scores.tsv.gz',vartype=['snps','indels']),

rule get_data:
   '''
   Get CADD 1.7 training data and scores
   '''
    output:
        expand(progress_dir + 'CADD.{vartype}.{label}.vcf.gz',label=['pos','neg'],vartype=['snps','indels']),
        expand(progress_dir + 'CADD.{vartype}.scores.tsv.gz',vartype=['snps','indels']),
    log:
        progress_dir + 'download.log'
    shell:
        r'''
        base_url='https://kircherlab.bihealth.org/download/CADD-development/v1.7/training_data/GRCh38/'

        wget -q -O - ${{base_url}}/humanDerived_InDels.vcf.gz > {progress_dir}/'CADD.indels.neg.vcf.gz'
        wget -q -O - ${{base_url}}/humanDerived_SNVs.vcf.gz > {progress_dir}/'CADD.snps.neg.vcf.gz'

        wget -q -O - ${{base_url}}/simulation_InDels.vcf.gz > {progress_dir}/'CADD.indels.pos.vcf.gz'
        wget -q -O - ${{base_url}}/simulation_SNVs.vcf.gz > {progress_dir}/'CADD.snps.pos.vcf.gz'

        base_url='https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/'

        wget -q -O - ${{base_url}}/whole_genome_SNVs.tsv.gz > {progress_dir}/'CADD.snps.scores.tsv.gz'
        wget -q -O - ${{base_url}}/gnomad.genomes.r4.0.indel.tsv.gz > {progress_dir}/'CADD.indels.scores.tsv.gz'
        '''

rule intersect_3utr_train:
    '''
    convert CADD training data to.bed format and intersect with 3'UTR regions bed
    '''
    input:
        vcf = progress_dir + 'CADD.{vartype}.{label}.vcf.gz',
        utr_bed = utr_bed,
    output:
        bed = progress_dir + 'CADD.3utr.{vartype}.{label}.bed',
    shell:
        r'''
        zcat {input.vcf}|sort -k1,1n -k2,2n|awk 'BEGIN{{OFS="\t"}}{{print $1,$2-1,$2,"GT="$4"/"$5}}' | \
        bedtools intersect -sorted -wo -b 'stdin' -a {input.utr_bed} |awk 'BEGIN{{OFS="\t"}}{{print "chr"$11,$12,$13,$14";seq_name="$4}}' > {output.bed}
        '''

rule tabix_scores:
    input:
        progress_dir + 'CADD.{vartype}.scores.tsv.gz'
    output:
        progress_dir + 'CADD.{vartype}.scores.tsv.gz.tbi'
    shell:
        r'''
        tabix -s1 -b2 -e2 {input}
        '''

rule intersect_3utr_scores:
    input:
        tsv = progress_dir + 'CADD.{vartype}.scores.tsv.gz',
        tbi = progress_dir + 'CADD.{vartype}.scores.tsv.gz.tbi',
        utr_bed = utr_bed,
    output:
        tsv = progress_dir + 'CADD.3utr.{vartype}.scores.tsv.gz',
        tbi = progress_dir + 'CADD.3utr.{vartype}.scores.tsv.gz.tbi',
    shell:
        r'''
        tabix {input.tsv} -R {input.utr_bed} |sed 's/^/chr/'|sort -k1,1 -k2,2n|bgzip -c > {output.tsv}
        tabix -s1 -b2 -e2 {output.tsv}
        '''
