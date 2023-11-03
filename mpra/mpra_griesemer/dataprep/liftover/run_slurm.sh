#!/bin/bash
#exclude servers with no AVX support (otherwise calling takes ages)!
#mutect2 log should have the line "Using CPU-supported AVX-512 instructions"
#if the line is absent, the node should also be excluded as AVX works somehow more slowly on these machines

source /home/icb/sergey.vilov/.bashrc
conda activate vale
rm -f slurm_logs/*
mkdir slurm_logs
snakemake -s Snakefile.py -k --restart-times 0 --rerun-incomplete --use-conda --latency-wait 180 --cluster-config cluster.yaml --cluster 'sbatch -p cpu_p \
--mem={cluster.mem} --time=2-00:00:00 --nice=10000 -o slurm_logs/%j.out' -j 200
#-x ibis216-010-0[68-71] \
