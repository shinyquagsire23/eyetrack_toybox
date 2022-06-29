#!/bin/zsh
(exit 69)
#echo $?
while [ $? -eq 69 ]
do
    python3 eye_net.py resume
done