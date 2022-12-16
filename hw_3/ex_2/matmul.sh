#!/bin/bash

#clear output file
echo -n "Matrix Dimension, CPU Kernel, GPU Kernel, Host to Device, Device to Host" > matmul.csv
echo "" >> matmul.csv

power=( 5 6 7 8 9 10 )

for i in ${power[@]}
do
    dim=$((2<<$i))
    echo $dim
    echo $(./matmul.out $dim $dim $dim 1) >> matmul.csv
done

