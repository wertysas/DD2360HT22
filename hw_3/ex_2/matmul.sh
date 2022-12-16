#!/bin/bash

#clear output file
echo -n "Matrix Dimension, CPU Kernel, GPU Kernel, Host to Device, Device to Host" > matmul.csv
echo "" >> matmul.csv

power=( 6 7 8 9 10 11 12 )

for i in ${power[@]}
do
    echo $(./matmul.out $(( 2<<$i )) $(( 2<<$i )) $(( 2<<$i )) $(( 2<<$i )) 1) >> matmul.csv
done


