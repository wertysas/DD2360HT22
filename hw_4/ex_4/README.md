## Finite Difference Methods for Heat Equation in 1D

single source which can be compiled with
```
nvcc -O3 -o finite_difference finite_difference.cu -lm -lcublas -lcusparse
```
and executed with the command
```
./finite_difference [number of grid points] [time steps]
```
