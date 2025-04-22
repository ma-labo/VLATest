#!/bin/zsh

models=(rt_1_x rt_1_400k rt_1_58k rt_1_1k octo-base octo-small openvla-7b)

tasks=(grasp move put-on put-in)


for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
      echo RUN "${model}" on "${task}" "with random lighting"
      PYTHONPATH=~/VLATest python3 run_fuzzer_w_lighting.py -m "${model}" -io /data/ -s 1682684216 -t "${task}"
  done
done