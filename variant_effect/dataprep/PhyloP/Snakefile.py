progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/prefiltered/PhyloP/' #output dir

utr_bed = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

rule all:
    input:
        expand(progress_dir + '{model}.bedGraph',model=['PhyloP-100way','PhyloP-241way']),
        expand(progress_dir + '{model}.3utr.scores.tsv.gz',model=['PhyloP-100way','PhyloP-241way']),

rule get_PhyloP_241way:
    output:
        progress_dir + 'PhyloP-241way.bigWig',
    shell:
        r'''
        wget -q -O - https://cgl.gi.ucsc.edu/data/cactus/241-mammalian-2020v2-hub/Homo_sapiens/241-mammalian-2020v2.bigWig > {progress_dir}/PhyloP-241way.bigWig
        '''

rule get_PhyloP_100way:
    output:
        progress_dir + 'PhyloP-100way.bigWig',
    shell:
        r'''
        wget -q -O - https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw > {progress_dir}/PhyloP-100way.bigWig
        '''

rule bigwig_to_bedgraph:
    input:
        bw = progress_dir + '{model}.bigWig',
    output:
        bg = progress_dir + '{model}.bedGraph',
    shell:
        r'''
        bigWigToBedGraph {input} {output}
        '''

rule bedgraph_to_tsv:
    input:
        bg = progress_dir + '{model}.bedGraph',
    output:
        tsv = progress_dir + '{model}.tsv.gz',
        tbi = progress_dir + '{model}.tsv.gz.tbi',
    shell:
        r'''
        cat {input} | awk '{{print $1 "\t" $2+1 "\t" $4 }}'|bgzip -c > {output.tsv}
        tabix -s1 -b2 -e2 {output.tsv}
        '''       


rule intersect_3utr_scores:
    input:
        tsv = progress_dir + '{model}.tsv.gz',
        tbi = progress_dir + '{model}.tsv.gz.tbi',
        utr_bed = utr_bed,
    output:
        tsv = progress_dir + '{model}.3utr.scores.tsv.gz',
        tbi = progress_dir + '{model}.3utr.scores.tsv.gz.tbi',
    shell:
        r'''
        tabix {input.tsv} -R {input.utr_bed} |sort -k1,1 -k2,2n|bgzip -c > {output.tsv}
        tabix -s1 -b2 -e2 {output.tsv}
        '''
