


progress_dir = '/s/project/mll/sergey/effect_prediction/MLM/griesemer/' #output dir

liftover_dir = '/s/project/mll/sergey/effect_prediction/tools/liftOver/'

rule all:
    input:
        progress_dir + 'liftover/varpos_GRCh38.bed',
        progress_dir + 'liftover/oligopos_GRCh38.bed',

rule extract_var_coords:
    '''
    Conver variant coordinates to 0-based format and write to BED12 file, ignore NaNs
    '''
    input:
        tsv = progress_dir + 'paper_supplementary/Oligo_Variant_Info.txt'
    output:
        bed = progress_dir + 'liftover/varpos_GRCh37.bed',
    shell:
        r'''
        tail -n +2 {input.tsv} | awk 'BEGIN{{OFS="\t"}}{{print $5,$9-1,$10,$11,$12,"0",$4}}'| grep  -v NA   \
        | sort -k1,1 -k2,2n | uniq | sed -e 's/^/chr/' -e 's/chrMT/chrM/'  > {output.bed}
        '''

rule extract_oligo_coords:
    '''
    Conver oligo coordinates to 0-based format and write to BED12 file, ignore NaNs
    '''
    input:
        tsv = progress_dir + 'paper_supplementary/Oligo_Variant_Info.txt'
    output:
        bed = progress_dir + 'liftover/oligopos_GRCh37.bed',
    shell:
        r'''
        tail -n +2 {input.tsv} |awk 'BEGIN{{OFS="\t"}}{{print $5,$6-1,$7,$3}}'| grep  -v NA | grep -v ',' \
        | sort -k1,1 -k2,2n | uniq | sed -e 's/^/chr/' -e 's/chrMT/chrM/'  > {output.bed}
        '''

rule liftover_vars:
    '''
    Liftover from GRCh37 to GRCh38
    '''
    input:
        bed = progress_dir + 'liftover/varpos_GRCh37.bed',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'liftover/varpos_GRCh38.bed',
        umap = temp(progress_dir + 'liftover/transcripts_canonocal_GRCh38.umap')
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''

rule liftover_oligo:
    '''
    Liftover from GRCh37 to GRCh38
    '''
    input:
        bed = progress_dir + 'liftover/oligopos_GRCh37.bed',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'liftover/oligopos_GRCh38.bed',
        umap = temp(progress_dir + 'liftover/transcripts_canonocal_GRCh38.umap')
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''
