
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

// Timer function
double time() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec*1e3 + (double)tp.tv_usec*1e-3);
}


int main(int argc, char **argv) {
  uint inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
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
  double t0 = time();
  for (uint i=0; i<inputLength; i++) {
    resultRef[i] =  hostInput1[i] + hostInput2[i];
  }
  double cpuTiming = time() - t0;

  // GPU memory allocation
  cudaMalloc(&deviceInput1, vsizeB);
  cudaMalloc(&deviceInput2, vsizeB);
  cudaMalloc(&deviceOutput, vsizeB);

  // Copy memory to the GPU
  t0 = time();
  cudaMemcpy(&deviceInput1, hostInput1, vsizeB, cudaMemcpyHostToDevice);
  cudaMemcpy(&deviceInput2, hostInput2, vsizeB, cudaMemcpyHostToDevice);
  double timeHostToDevice = time() - t0;
  
  // Initialize the 1D grid and block dimensions
  uint blockSize = TPB;
  uint gridSize  = (inputLength+blockSize-1) / blockSize;

  // Launch the GPU Kernel
  t0 = time();
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double kernelTime = time() - t0;

  // Copy the GPU memory back to the CPU
  t0 = time();
  cudaMemcpy(&hostOutput, &deviceOutput, vsizeB, cudaMemcpyDeviceToHost);
  double timeDeviceToHost = time() - t0;

  // Compare the output with the reference
  DataType max_diff = 1e-7;
  for (uint i=0; i<inputLength; i++) {
    if (abs(hostOutput[i]-resultRef[i])<1e-7) {
      printf("Error results differ more than maximum value (>%f)\n", max_diff);
      printf("Host Calculated Value: %f\n", resultRef[i]);
      printf("Device Calculated Value: %f\n", hostOutput[i]);
    }
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



  // Measurement prints
  if (formattedPrint) {
    printf("%d, %11.8f, %11.8f, %11.8f, %11.8f\n", inputLength, cpuTiming, kernelTime, timeHostToDevice, timeDeviceToHost);
  }
  else {
    printf("Vector length: %d\n", inputLength);
    printf("CPU vector addition time: %f\n", cpuTiming);
    printf("GPU vector addition time: %f\t(excl. data transfer)\n", kernelTime);
    printf("GPU vector addition time: %f\t(incl. data transfer)\n", kernelTime+timeHostToDevice+timeDeviceToHost);
    printf("Host to Device data transfer time: %f\n", timeHostToDevice);
    printf("Device to Host data transfer time: %f\n", timeDeviceToHost);
  }
  return 0;
}

