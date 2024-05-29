#!/bin/bash

resume_arg=
chkpnt_dir="../checkpoints/ttnet_3rd_phase"
last_chkpnt=`/bin/ls -t ${chkpnt_dir} 2>/dev/null | head -n1`
resume_path="${chkpnt_dir}/${last_chkpnt}"
if [ -f "${resume_path}" ]; then
  echo "Resume from ${resume_path}"
  resume_arg="--resume_path ${resume_path}"
fi

phase2_dir="../checkpoints/ttnet_2nd_phase"
phase2_chkpnt=`/bin/ls -t ${phase2_dir} 2>/dev/null | head -n1`
pretrained_path="${phase2_dir}/${phase2_chkpnt}"
if [ ! -f "${pretrained_path}" ]; then
  echo "No last pretrained_path, exit"
  exit 1
fi

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_3rd_phase' \
  --no-val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.2 \
  --no_cuda \
  --global_weight 1. \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ${pretrained_path} \
  --smooth-labelling ${resume_arg}