# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2

all: bfs bfs_offload

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs.out

bfs_offload: bfs.cpp
	$(CC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload.out

clean:
	rm -f bfs.out bfs_offload result.txt
