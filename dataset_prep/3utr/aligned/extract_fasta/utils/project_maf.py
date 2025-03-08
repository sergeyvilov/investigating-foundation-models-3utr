#########################################################################################
#Project maf alignment onto the reference genome
#
#Input: .maf.gz file
#Output: projected.maf.gz file
#
#The reference sequence must be on the top of each alignment block!
#########################################################################################

import sys
import re
import gzip

input_maf_gz = sys.argv[1] #input .maf.gz
output_maf_gz = sys.argv[2] #output .maf.gz

def project_block(block, maf_out):
    '''
    Split an initial alignment block into multiple blocks, for each region between insertions in the reference sequence
    '''
    
    block_header = block[0] #block header
    
    ref_seq = block[1].split()[-1] #reference sequence, must be on the top of the block
    
    all_seq = block[1:] #all sequences, including reference 
    
    split_groups = [x.span() for x in list(re.finditer('[^-]+',ref_seq))] #in the reference seq, find regions separated by insertions
    
    for split_start, split_end  in split_groups: #loop over all regions: each region will lead to a new alignment block
        
        maf_out.write(block_header) #same header for all regions (subblocks)
        
        for row in all_seq:
            
            row_type, contig_name, seq_start, _, strand, fragment_len, seq = row.split()
            
            matches_before_split = len(list(filter(lambda x:x!='-',seq[:split_start]))) #number of maches to the left of the split
            
            seq_start = int(seq_start) + matches_before_split #change the start coordinate of the current sequence
            
            seq = seq[split_start:split_end] #part of the current sequence corresponding to the region of the reference sequence btw insertions
            
            match_len = len(list(filter(lambda x:x!='-',seq))) #number of matches in the chosen part of the sequence
            
            if match_len!=0: #if there're matches, print 
                maf_out.write('\t'.join((row_type,contig_name,str(seq_start),str(match_len),strand,fragment_len,seq))+'\n')
        
        maf_out.write('\n')
        
with gzip.open(input_maf_gz,'rt') as maf_in:
    
    with gzip.open(output_maf_gz,'wt') as maf_out: 
        
        line = maf_in.readline() 
        
        while line.startswith('##'):
            #maf header
            maf_out.write(line)
            line = maf_in.readline() 
            
        #now line is the block header 
        
        current_alignment_block = [line] #add the block header
        
        for line in maf_in: 
            
            while not line.startswith('a'):
                
                if line.startswith('s'):
                    current_alignment_block.append(line)
                    
                line = maf_in.readline()
                
                if line=='': #EOF
                    break
                    
            project_block(current_alignment_block, maf_out)
            
            current_alignment_block = [line] #start new alignment block