oligo_variant_info = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/mpra/griesemer_2021/paper_supplementary/Oligo_Variant_Info.txt'

progress_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/mpra/griesemer_2021/liftover/' #output dir

liftover_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/tools/liftOver/'

rule all:
    input:
        progress_dir + 'varpos_GRCh38.bed',
        progress_dir + 'oligopos_GRCh38.bed',

rule extract_var_coords:
    '''
    Convert variant coordinates to 0-based format and write to BED12 file, ignore NaNs
    '''
    input:
        tsv = oligo_variant_info
    output:
        bed = temp(progress_dir + 'varpos_GRCh37.bed'),
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
        tsv = oligo_variant_info
    output:
        bed = temp(progress_dir + 'oligopos_GRCh37.bed'),
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
        bed = progress_dir + 'varpos_GRCh37.bed',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'varpos_GRCh38.bed',
        umap = temp(progress_dir + 'transcripts_canonocal_GRCh38.umap')
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
        bed = progress_dir + 'oligopos_GRCh37.bed',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'oligopos_GRCh38.bed',
        umap = temp(progress_dir + 'transcripts_canonocal_GRCh38.umap')
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''
