#!/bin/bash 
mkdir -p log-qsub
qsub -l 'gpu=1' \
  -cwd \
  -j y \
  -v PATH \
  -S /bin/zsh \
  -o $1.qsub.log \
  $1

