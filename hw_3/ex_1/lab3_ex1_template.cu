
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#include <random>

#define DataType double
#define TPB 256

using uint = unsigned int;

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, uint len) {
  const uint idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < len) {
    out[idx] =  in1[idx] + in2[idx];
  }
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  uint inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  // Input length reading
  inputLength = int(*argv[1]);
  printf("The input length is %d\n", inputLength);
  
  // Host memory allocation
  size_t vsizeB =  inputLength*sizeof(double);
  hostInput1 = (DataType*) malloc(vsizeB);
  hostInput2 = (DataType*) malloc(vsizeB);
  hostOutput = (DataType*) malloc(vsizeB);
  resultRef = (DataType*) malloc(vsizeB);
  
  // Initialisation of host arrays to random values
  std::default_random_engine e{};
  std::uniform_real_distribution<DataType> d{-1.0, 1.0};
  for (uint i=0; i<inputLength; i++) { hostInput1[i] = d(e); }
  for (uint i=0; i<inputLength; i++) { hostInput2[i] = d(e); }
  
  // Computation of reference result on host
  for (uint i=0; i<inputLength; i++) {
    resultRef[i] =  hostInput1[i] + hostInput2[i];
  }
    

  // GPU memory allocation
  cudaMalloc(&deviceInput1, vsizeB);
  cudaMalloc(&deviceInput2, vsizeB);
  cudaMalloc(&deviceOutput, vsizeB);

  // Copy memory to the GPU
  cudaMemcpy(&deviceInput1, hostInput1, vsizeB, cudaMemcpyHostToDevice);
  cudaMemcpy(&deviceInput2, hostInput2, vsizeB, cudaMemcpyHostToDevice);

  // Initialize the 1D grid and block dimensions
  uint blockSize = TPB;
  uint gridSize  = (inputLength+blockSize-1) / blockSize;

  // Launch the GPU Kernel
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  // Copy the GPU memory back to the CPU
  cudaMemcpy(&hostOutput, &deviceOutput, vsizeB, cudaMemcpyDeviceToHost);

  // Compare the output with the reference
  DataType max_diff = 1e-7;
  for (uint i=0; i<inputLength; i++) {
    if (abs(hostOutput[i]-resultRef[i])<1e-7) {
      printf("Error results differ more than maximum value (>%f)\n", max_diff);
      printf("Host Calculated Value: %f\n", resultRef[i]);
      printf("Device Calculated Value: %f\n", hostOutput[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
