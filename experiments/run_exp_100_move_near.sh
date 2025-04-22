#!/bin/zsh

models=(rt_1_x rt_1_400k rt_1_58k rt_1_1k octo-base octo-small openvla-7b)

datasets=(t-move_n-100_o-0_s-3225323079.json t-move_n-100_o-1_s-474914166.json t-move_n-100_o-2_s-1227205386.json t-move_n-100_o-3_s-1497023353.json t-move_n-100_o-4_s-619210020.json)

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo RUN "${model}" on "${data}"
    PYTHONPATH=~/VLATest python3 run_fuzzer.py -s 2024 -m "${model}" -d ../data/"${data}" -io /data/
  done
done
