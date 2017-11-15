#!/bin/bash -v
mkdir -p log-qsub
qsub -l 'hostname=b1[123456789]*,gpu=1' \
  -cwd \
  -j y \
  -v PATH \
  -S /bin/zsh \
  -o ./log-qsub/$1.log \
  $1
