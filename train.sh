#!/bin/zsh
#rm -rf logs
mkdir -p logs
mkdir -p logs/fit
python3 eye_net.py train
while [ $? -eq 69 ]
do
    python3 eye_net.py resume
done