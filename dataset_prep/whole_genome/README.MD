Build whole genome Zoonomia dataset to train DNABERT2-ZOO

1. Extract all Zoonomia species from the .hal archive: `sbatch --array=1-241%10 extract_train.sh`.
2. Merge species sequences into a single FASTA file: `sbatch extract_train.sh`.
3. Generate shuffled splits: `sbatch get_splits.sh`.
We split long contigs into chunks of CHUNK_LEN=100000 bp with an overlap of OVERLAP_BP=50 bp  (see `get_splits.py`).
In total, we produce N_SPLITS = 1000 splits spanning all Zoonomia genomes.
4. Get FASTA files for each split: `sbatch --array=0-999%30 collect_splits.sh`.
5. Merge random splits into a single txt file: `sbatch merge_splits.sh`.
