/*
* Naive Cuda MatMul, matrices are stored in arrays C-style, i.e. row-major format
* for an m x n matrix stored in array  A:  a_ij = A[n*i + j]
*/ 

#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType double
#define TPB 256

using uint = unsigned int;

// Timer function (milliseconds)
double time() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec*1e3 + (double)tp.tv_usec*1e-3);
}

// Compute C = A * B (naive implementation no shared memory or tiling)
__global__ void gemm(DataType *A, DataType *B, DataType *C, uint numARows,
                      uint numAColumns, uint numBRows, uint numBColumns){
  //  idxes defining c_ij to be computed by thread
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  uint j = blockIdx.y*blockDim.y + threadIdx.y;
  //  return if we're out of the array boundary
  if ( i >= numARows || j >= numBColumns ) { return;}
  DataType cij = 0.0;
  for (uint k=0; k<numBRows; k++) {
    cij += A[i*numAColumns + k]*B[k*numBColumns + j];
  }
  C[i*numBColumns + j] = cij;
}


int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB; DataType *deviceC;
  uint numARows;    // number of rows in the matrix A
  uint numAColumns; // number of columns in the matrix A
  uint numBRows;    // number of rows in the matrix B
  uint numBColumns; // number of columns in the matrix B
  uint numCRows;
  uint numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  if (!(argc == 4 || argc == 5)) {
    printf("Error wrong format! Correct Usage :\t ./matmul [nrows matrix 1] [ncols matrix 1] \
            [ncols matrix 2]\n \
            modes:\n\t0 (default output)\n\t1 (csv formatted output for measurements)\n");
    return -1;
  }
  
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numAColumns;


  bool formattedPrint = false;
  if (argc==5 && atoi(argv[4])==1) {
    formattedPrint = true;
  }
  else {
    printf("Input matrix dim A: (%d x %d)\tB:(%d x %d)\tC: (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  }
  
  // Host memory for input and output
  size_t sizeA =  numARows*numAColumns*sizeof(DataType);
  size_t sizeB =  numBRows*numBColumns*sizeof(DataType);
  size_t sizeC =  numCRows*numCColumns*sizeof(DataType);
  hostA = (DataType*) malloc(sizeA);
  hostB = (DataType*) malloc(sizeB);
  hostC = (DataType*) malloc(sizeC);
  resultRef = (DataType*) malloc(sizeC);
  
  // Initialisation of A and B to random numbers
  std::default_random_engine e{};
  std::uniform_real_distribution<DataType> d{-1.0, 1.0};
  for (uint i=0; i<numARows*numAColumns; i++) { hostA[i] = d(e); }
  for (uint i=0; i<numBRows*numBColumns; i++) { hostB[i] = d(e); }

  // Reference result computation
  double t0 = time();
  for (uint i=0; i<numARows; i++) {
    for (uint j=0; j<numBColumns; j++) {
      resultRef[numCColumns*i + j] = 0.0;
      for (uint k=0; k<numAColumns; k++) {
        resultRef[numCColumns*i + j] += hostA[numAColumns*i + k]*hostB[numBColumns*k + j];
      }
    }
  }
  double cpuTiming = time() - t0;
  
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, sizeA);
  cudaMalloc(&deviceB, sizeB);
  cudaMalloc(&deviceC, sizeC);

  //@@ Insert code to below to Copy memory to the GPU here
  t0 = time();
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
  double timeHostToDevice = time() - t0;


  //@@ Initialize the grid and block dimensions here we use (1,1) threadblocks
  uint blockSize = 32;
  dim3 dimGrid((numCRows+blockSize-1)/blockSize,(numCColumns+blockSize-1)/blockSize );
  dim3 dimBlock(blockSize, blockSize);

  //@@ Launch the GPU Kernel here
  t0 = time(); 
  gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double kernelTime = time() - t0;

  //@@ Copy the GPU memory back to the CPU here
  t0 = time();
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  double timeDeviceToHost = time() - t0;
  

  //@@ Insert code below to compare the output with the reference
  DataType max_diff = 1e-7;
  for (uint i=0; i<numCRows*numCColumns; i++) {
    if (abs(hostC[i]-resultRef[i])>1e-7) {
      printf("Error results differ more than maximum value (>%f)\n", max_diff);
      printf("Host Calculated Value: %f\n", resultRef[i]);
      printf("Device Calculated Value: %f\n", hostC[i]);
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  
  // Measurement prints (we assume square matrices for this)
  if (formattedPrint) {
    printf("%d, %11.8f, %11.8f, %11.8f, %11.8f\n", numARows, cpuTiming, kernelTime, timeHostToDevice, timeDeviceToHost);
  }
  else {
    printf("CPU gemm time: %f\n", cpuTiming);
    printf("GPU gemm time: %f\t(excl. data transfer)\n", kernelTime);
    printf("GPU gemm time: %f\t(incl. data transfer)\n", kernelTime+timeHostToDevice+timeDeviceToHost);
    printf("Host to Device data transfer time: %f\n", timeHostToDevice);
    printf("Device to Host data transfer time: %f\n", timeDeviceToHost);
  }

  return 0;
}
