################################################################################################################
# Get reference and alt allelic depths of flanking variants for each variant in a VCF file
#
#
# python get_flanking.py vcf_name min_gnomAD_AF flanking_region_length
#
# vcf_name: single-sample VCF file with all possible variants (Mutect2 output in tumor-only mode)
# VCF file should be annotated with gnomAD allele frequency (gnomAD_AF) in the INFO field
# min_gnomAD_AF: minimal gnomAD AF to consider that a variant is germline (to detect flanking variants)
# flanking_region_length: how many bases the flanking region spans
# to the left and (or to the righ) of a candidate variant
################################################################################################################

import numpy as np
import pandas as pd
import re
import os
import sys

MIN_AD = 5 #minimal AD for a flanking variant
N_FLANKING_EACH_SIDE = 2 #number of flanking variants to consider at each side

def AD_from_format(row):
    '''
    Get allelic depth from Strelka format string,
    see https://github.com/Illumina/strelka/tree/master/docs/userGuide#somatic
    '''

    if ',' in row.ref or ',' in row.alt:
        #biallelic variant
        return -1, -1

    counts = {k:int(v.split(',')[0]) for k, v in zip(row.schema.split(':'),row.format.split(':')) if ',' in v}

    if 'TAR' in row.schema:
        return counts['TAR'], counts['TIR']
    else:
        return counts[row.ref+'U'], counts[row.alt+'U']

def get_flanking(variants_pos, germline_df):
    '''
    Get flanking variants for each candidate variant position

    variants_pos: positions of candidate variants
    germline_df: (likely germline) variants, that can be flanking variants for positions in variants_pos
    '''

    germline_df = germline_df.sort_values(by='pos')

    L = len(germline_df)

    AD = germline_df[['AD_ref','AD_alt']].values.astype(int)

    for variant_pos in variants_pos:

        min_left_idx = np.searchsorted(germline_df.pos, variant_pos-flanking_region_length)
        max_right_idx = np.searchsorted(germline_df.pos, variant_pos+flanking_region_length)
        variant_idx = np.searchsorted(germline_df.pos, variant_pos)

        flanking_list = [-1 for _ in range(N_FLANKING_EACH_SIDE*2*2)]

        flanking_left = AD[max(variant_idx-N_FLANKING_EACH_SIDE,min_left_idx,0):variant_idx][::-1]

        for idx, (AD_ref, AD_alt) in enumerate(flanking_left):
            flanking_list[idx*2:idx*2+2] = AD_ref, AD_alt

        if variant_idx<L and germline_df.iloc[variant_idx].pos==variant_pos: #avoid overlap between candodate variant and its flanking variant
            variant_idx+=1

        flanking_right = AD[variant_idx:min(variant_idx+N_FLANKING_EACH_SIDE,max_right_idx)]

        for idx, (AD_ref, AD_alt) in enumerate(flanking_right):
            flanking_list[idx*2+N_FLANKING_EACH_SIDE*2:idx*2+N_FLANKING_EACH_SIDE*2+2] = AD_ref, AD_alt

        print('|'.join([str(x) for x in flanking_list]))

min_gnomAD_AF = float(sys.argv[2])

flanking_region_length = int(sys.argv[3])

vcf = pd.read_csv(sys.argv[1], comment='#', sep='\t', usecols=[0,1,3,4,7,8,10], names=['chrom', 'pos', 'ref', 'alt', 'info', 'schema', 'format'], dtype={'chrom':str})

vcf['Filter'] = 'PASS' #all variants can potentially be flanking variants

#print('Filtering out all gnomAD variants with AF<',min_gnomAD_AF)

vcf['gnomAD_AF'] = vcf['info'].apply(lambda x: re.search('gnomAD_AF=([^;]*)',x).groups(1)[0] if 'gnomAD_AF' in x else 0).astype(float)

vcf.loc[vcf.gnomAD_AF < min_gnomAD_AF, 'Filter'] = 'no_germ_evidence'

#print('Filtering out all variants with AD<',MIN_AD)

vcf['AD_ref'], vcf['AD_alt'] = zip(*vcf[['ref','alt','schema','format']].apply(lambda x: AD_from_format(x), axis=1).tolist())

vcf.loc[(vcf.AD_ref<MIN_AD) | (vcf.AD_alt<MIN_AD), 'Filter'] = 'low_AD'

vcf = vcf[['chrom', 'pos', 'AD_ref', 'AD_alt', 'Filter']] #only variants with Filter=PASS can be flanking variants

#print('Looking for flanking variants')

for chrom in vcf.chrom.drop_duplicates():
    chrom_df = vcf[vcf.chrom==chrom]
    get_flanking(chrom_df.pos.values, chrom_df[chrom_df['Filter']=='PASS'])
