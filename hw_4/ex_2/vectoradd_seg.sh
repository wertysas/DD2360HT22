#!/bin/bash

#clear output file
echo -n "Segment Size, GPU Kernel Streamed (ms)" > vectoradd_seg.csv
echo "" >> vectoradd_seg.csv

power=( 9 10 11 12 13 14 15)

for i in ${power[@]}
do
    echo $(./vectoradd.out 1048576 $(( 2<<$i ))) >> vectoradd.csv
done

