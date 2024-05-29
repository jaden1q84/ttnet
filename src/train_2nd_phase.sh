#!/bin/bash

resume_arg=
chkpnt_dir="../checkpoints/ttnet_2nd_phase"
last_chkpnt=`/bin/ls -t ${chkpnt_dir} 2>/dev/null | head -n1`
resume_path="${chkpnt_dir}/${last_chkpnt}"
if [ -f "${resume_path}" ]; then
  echo "Resume from ${resume_path}"
  resume_arg="--resume_path ${resume_path}"
fi

phase1_dir="../checkpoints/ttnet_1st_phase"
phase1_chkpnt=`/bin/ls -t ${phase1_dir} 2>/dev/null | head -n1`
pretrained_path="${phase1_dir}/${phase1_chkpnt}"
if [ ! -f "${pretrained_path}" ]; then
  echo "No last pretrained_path, exit"
  exit 1
fi

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_2nd_phase' \
  --no-val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --no_cuda \
  --global_weight 0. \
  --seg_weight 0. \
  --event_weight 2. \
  --local_weight 1. \
  --pretrained_path ${pretrained_path} \
  --overwrite_global_2_local \
  --freeze_seg \
  --freeze_global \
  --smooth-labelling ${resume_arg}