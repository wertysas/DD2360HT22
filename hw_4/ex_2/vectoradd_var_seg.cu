
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <random>
#include <assert.h>

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

__global__ void vecAddNonStreamed(DataType *in1, DataType *in2, DataType *out, uint len) {
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

  // Input length reading
  if (!(argc == 2 || argc ==3)) {
    printf("Error wrong format! Correct Usage :\t ./vectoradd [vector length] [segment length]\n \
            vector length:\tpositive integer\n \
            segment length:\tmust divide vector length\n");
    return -1;
  }
  inputLength = atoi(argv[1]);
  int segmentLength = atoi(argv[2]);

  // Host memory allocation
  DataType *resultRef;
  size_t vsizeB =  inputLength*sizeof(DataType);
  resultRef = (DataType*) malloc(vsizeB);
  
   
  // GPU memory allocation
  DataType *deviceInput1, *deviceInput2, *deviceOutput;
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
  for (uint i=0; i<inputLength; i++) {
    resultRef[i] =  pinnedInput1[i] + pinnedInput2[i];
  }

  // Create and initialise streams
  int nStreams = (inputLength+segmentLength-1)/segmentLength;
  cudaStream_t streams[nStreams];
  int streamSize = (inputLength + nStreams - 1) / nStreams;
  assert(streamSize = segmentLength);
  int lastStreamSize = inputLength - streamSize*(nStreams-1);
  int streamBytes = sizeof(DataType) * streamSize;
  int lastStreamBytes = sizeof(DataType) * lastStreamSize;
  int offset;
  for (int i=0; i<nStreams; i++) {
    checkCuda( cudaStreamCreate(&streams[i]) );
  }
  
  // Initialize the 1D grid and block dimensions
  uint blockSize = TPB;
  uint gridSize  = (streamSize+blockSize-1) / blockSize;

  // Copy and kernel execution
  double t0 = time();
  for (int i=0; i<nStreams-1; i++) {
    offset= i*streamSize;
    checkCuda( cudaMemcpyAsync(&deviceInput1[offset], &pinnedInput1[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]) );
    checkCuda( cudaMemcpyAsync(&deviceInput2[offset], &pinnedInput2[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]) );
    vecAdd<<<gridSize, blockSize, 0, streams[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);
    checkCuda( cudaMemcpyAsync(&pinnedOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]) );
  }
  int i = nStreams-1;
  offset= i*streamSize;
  gridSize = (lastStreamSize+blockSize-1) / blockSize;
  checkCuda( cudaMemcpyAsync(&deviceInput1[offset], &pinnedInput1[offset], lastStreamBytes, cudaMemcpyHostToDevice, streams[i]) );
  checkCuda( cudaMemcpyAsync(&deviceInput2[offset], &pinnedInput2[offset], lastStreamBytes, cudaMemcpyHostToDevice, streams[i]) );
  vecAdd<<<gridSize, blockSize, 0, streams[nStreams-1]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], lastStreamSize);
  checkCuda( cudaMemcpyAsync(&pinnedOutput[offset], &deviceOutput[offset], lastStreamBytes, cudaMemcpyDeviceToHost, streams[i]) );;
   
  for (int i=0; i<nStreams; i++) {
    checkCuda( cudaStreamSynchronize(streams[i]) );
  }
  // Timing
  double gpuTiming = time() - t0;
  

  // Compare the output with the reference
  DataType max_diff = 1e-7;
  for (uint i=0; i<inputLength; i++) {
    if (abs(pinnedOutput[i]-resultRef[i])>1e-7) {
      printf("Index: %d, Error results differ more than maximum value (>%f)\n", i, max_diff);
      //printf("Host Calculated Value: %f\n", resultRef[i]);
      //printf("Device Calculated Value: %f\n", pinnedOutput[i]);
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
  printf("%d, %11.8f\n", segmentLength, gpuTiming);
  return 0;
}

