#!/bin/bash

#################################################################
# wrapper for multi-gpu training                                #
# kills all processes when an error in the log file is reported #
#                                                               #
#################################################################

errorfile=parallel.log

submit_res=$(sbatch ./test.sh)

echo $submit_res

job_id=$(echo $submit_res|cut -d" " -f4)

function finish {
  scancel $job_id
}

trap finish EXIT #cancel job on forced exit

while true; do
  sleep 120 #interval between checks, s
  running_jobs=( $(squeue --me -o "%i"|tail -n+2) ) #get all currently running jobs
  if [[ ${running_jobs[@]} =~ $job_id ]]; then
    grep "Exited\|error" $errorfile && scancel $job_id && exit 1 #if error in job output, kill all job processes
  else
    exit 0 #if job finished itself, just exit
  fi
done
