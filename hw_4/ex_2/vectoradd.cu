
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#include <random>

#define DataType double
#define TPB 256

using uint = unsigned int;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, uint len) {
  const uint idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < len) {
    out[idx] =  in1[idx] + in2[idx];
  }
}

// Timer function
double time() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec*1e3 + (double)tp.tv_usec*1e-3);
}


int main(int argc, char **argv) {
  uint inputLength;
  DataType *resultRef;
  DataType *deviceInput1; DataType *deviceInput2;
  DataType *deviceOutput;

  // Input length reading
  if (!(argc == 2 || argc ==3)) {
    printf("Error wrong format! Correct Usage :\t ./vectoradd [vector length] [mode]\n \
            vector length:\n\tpositive integer\n \
            modes:\n\t0 (default output)\n\t1 (csv formatted output for measurements)\n");
    return -1;
  }
  inputLength = atoi(argv[1]);
  bool formattedPrint = false;
  if (argc==3 && atoi(argv[2])==1) {
    formattedPrint = true;
  }

  // Host memory allocation
  size_t vsizeB =  inputLength*sizeof(DataType);
  resultRef = (DataType*) malloc(vsizeB);
  
   
  // GPU memory allocation
  cudaMalloc(&deviceInput1, vsizeB);
  cudaMalloc(&deviceInput2, vsizeB);
  cudaMalloc(&deviceOutput, vsizeB);

  // Pinned Memory allocation
  DataType *pinnedInput1, *pinnedInput2, *pinnedOutput;
  checkCuda(cudaMallocHost( (void**) &pinnedInput1, vsizeB) );
  checkCuda(cudaMallocHost( (void**) &pinnedInput2, vsizeB) );
  checkCuda(cudaMallocHost( (void**) &pinnedOutput, vsizeB) );
  
  // Initialisation of pinned host arrays to random values
  std::default_random_engine e{};
  std::uniform_real_distribution<DataType> d{-1.0, 1.0};
  for (uint i=0; i<inputLength; i++) { pinnedInput1[i] = d(e); }
  for (uint i=0; i<inputLength; i++) { pinnedInput2[i] = d(e); }
  
  // Computation of reference result on host
  double t0 = time();
  for (uint i=0; i<inputLength; i++) {
    resultRef[i] =  pinnedInput1[i] + pinnedInput2[i];
  }
  double cpuTiming = time() - t0;

  // Create and initialise streams
  const int nStreams = 4;
  cudaStream_t streams[nStreams];
  const int streamSize = (vsizeB + nStreams - 1) / nStreams;
  const int lastStreamSize = streamSize*nStreams-vsizeB;
  int offset;
  for (int i=0; i<nStreams; i++) {
    checkCuda( cudaStreamCreate(&streams[i]) );
  }
  
  // Initialize the 1D grid and block dimensions
  uint blockSize = TPB;
  uint gridSize  = (inputLength+blockSize-1) / blockSize;

  // Copy and kernel execution
  t0 = time();
  for (int i=0; i<nStreams-1; i++) {
    offset= i*streamSize;
    checkCuda( cudaMemcpyAsync(&deviceInput1[offset], &pinnedInput1[offset], streamSize, cudaMemcpyHostToDevice, streams[i]) );
    checkCuda( cudaMemcpyAsync(&deviceInput2[offset], &pinnedInput2[offset], streamSize, cudaMemcpyHostToDevice, streams[i]) );
    vecAdd<<<gridSize, blockSize, 0, streams[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);
    checkCuda( cudaMemcpyAsync(&pinnedOutput[offset], &deviceOutput[offset], streamSize, cudaMemcpyDeviceToHost, streams[i]) );
  }
  int i = nStreams-1;
  offset= i*streamSize;
  checkCuda( cudaMemcpyAsync(&deviceInput1[offset], &pinnedInput1[offset], lastStreamSize, cudaMemcpyHostToDevice, streams[i]) );
  checkCuda( cudaMemcpyAsync(&deviceInput2[offset], &pinnedInput2[offset], lastStreamSize, cudaMemcpyHostToDevice, streams[i]) );
  vecAdd<<<gridSize, blockSize, 0, streams[nStreams-1]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], lastStreamSize);
  checkCuda( cudaMemcpyAsync(&pinnedOutput[offset], &deviceOutput[offset], lastStreamSize, cudaMemcpyDeviceToHost, streams[i]) );

  // Copy memory to the GPU
  double vectorAddTiming = time() - t0;
  

  // Compare the output with the reference
  DataType max_diff = 1e-7;
  for (uint i=0; i<inputLength; i++) {
    if (abs(pinnedOutput[i]-resultRef[i])>1e-7) {
      printf("Error results differ more than maximum value (>%f)\n", max_diff);
      printf("Host Calculated Value: %f\n", resultRef[i]);
      printf("Device Calculated Value: %f\n", pinnedOutput[i]);
    }
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  cudaFreeHost(pinnedInput1);
  cudaFreeHost(pinnedInput2);
  cudaFreeHost(pinnedOutput);


  // Measurement prints
  if (formattedPrint) {
    printf("%d, %11.8f, %11.8f\n", inputLength, cpuTiming, vectorAddTiming);
  }
  else {
    printf("Vector length: %d\n", inputLength);
    printf("CPU vector addition time: %f\n", cpuTiming);
    printf("GPU vector addition time: %f\t(incl. data transfer)\n", vectorAddTiming);
  }
  return 0;
}
