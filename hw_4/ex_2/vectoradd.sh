#!/bin/bash

#clear output file
echo -n "Vector Size, CPU Kernel, GPU Kernel, Host to Device, Device to Host" > vectoradd.csv
echo "" >> vectoradd.csv

power=( 9 10 11 12 13 14 15 16 17 18 19 )

for i in ${power[@]}
do
    echo $(./vectoradd.out $(( 2<<$i )) 1) >> vectoradd.csv
done

