#!/bin/bash

resume_arg=
chkpnt_dir="../checkpoints/ttnet_1st_phase"
last_chkpnt=`/bin/ls -t ${chkpnt_dir} 2>/dev/null | head -n1`
resume_path="${chkpnt_dir}/${last_chkpnt}"
if [ -f "${resume_path}" ]; then
  echo "Resume from ${resume_path}"
  resume_arg="--resume_path ${resume_path}"
fi

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_1st_phase' \
  --no-val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --no_cuda \
  --global_weight 5. \
  --seg_weight 1. \
  --no_local \
  --no_event \
  --smooth-labelling ${resume_arg}