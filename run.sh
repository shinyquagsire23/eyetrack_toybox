#!/bin/zsh
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit" #--tf_xla_auto_jit=2 
export XLA_FLAGS="--xla_cpu_use_acl"

python3 eye_net.py run