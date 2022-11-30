#makefile

openmp: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm 


icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


openmp_clang: ex_particle_OPENMP_seq.c
	clang -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm  "-L/opt/homebrew/opt/llvm/lib"

clean:
	rm particle_filter
