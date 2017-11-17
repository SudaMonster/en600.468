#!/bin/bash 
mkdir -p log-qsub
qsub -l 'gpu=1,hostname=b1[123456789]*' \
  -cwd \
  -j y \
  -v PATH \
  -S /bin/zsh \
  -o ./log-qsub/$1.qsub.log \
  $1

