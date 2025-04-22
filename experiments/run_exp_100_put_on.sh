#!/bin/zsh

models=(rt_1_x rt_1_400k rt_1_58k rt_1_1k octo-base octo-small openvla-7b)

datasets=(t-put-on_n-100_o-0_s-3108960151.json t-put-on_n-100_o-1_s-4192256783.json t-put-on_n-100_o-2_s-3259027790.json t-put-on_n-100_o-3_s-274345646.json t-put-on_n-100_o-4_s-3571751632.json)

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo RUN "${model}" on "${data}"
    PYTHONPATH=~/VLATest python3 run_fuzzer.py -s 2024 -m "${model}" -d ../data/"${data}" -io /data/
  done
done
