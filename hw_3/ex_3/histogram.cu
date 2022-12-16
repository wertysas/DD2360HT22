
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 256

using uint = unsigned int;

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;   // index of current thread (out of all threads)
  
  // Creating shared array of bins and initialising to 0
  __shared__ uint shared_bins[NUM_BINS];

  if (TPB > NUM_BINS) {
    if (threadIdx.x < NUM_BINS) { shared_bins[threadIdx.x] = 0;}
  }
  else {  // NUM_BINS < TPB we let the first threads initialise shared bin array
    for (uint i=idx; i<NUM_BINS; i+=TPB) {
      shared_bins[i] = 0;
    }
  }
  __syncthreads();

  // Atomic adds to shared memory histogram
  if (idx < num_elements) { atomicAdd(&shared_bins[input[idx]], 1); }
  __syncthreads();
  
  // Atomic update of global memory histogram
  if (TPB > NUM_BINS) {
    if (threadIdx.x < NUM_BINS) {
      atomicAdd(&bins[threadIdx.x], shared_bins[threadIdx.x]);
    }
  }
  else {  // NUM_BINS < TPB we let the first threads initialise shared bin array
    for (uint i=idx; i<NUM_BINS; i+=TPB) {
      atomicAdd(&bins[i], shared_bins[i]);
    }
  }

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;   // index of current thread (out of all threads)
  if (idx < NUM_BINS) {
    if (bins[idx] > 127) { bins[idx] = 127; }
  }
}


int main(int argc, char **argv) {
  
  uint inputLength;
  uint *hostInput;
  uint *hostBins;
  uint *resultRef;
  uint *deviceInput;
  uint *deviceBins;

  //@@ Insert code below to read in inputLength from args
  
  if (!(argc == 2)) {
    printf("Error wrong format! Correct Usage:\t ./histogram [array length]\n");
    return -1;
  }
  
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  size_t inputSizeB =  inputLength*sizeof(uint);
  size_t binSizeB = NUM_BINS*sizeof(uint);
  hostInput = (uint*) malloc(inputSizeB);
  resultRef = (uint*) malloc(sizeof(uint)*NUM_BINS);
  hostBins = (uint*) malloc(sizeof(uint)*NUM_BINS);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::default_random_engine e{};
  std::uniform_int_distribution<uint> d{0, NUM_BINS-1};
  for (uint i=0; i<inputLength; i++) { hostInput[i] = d(e); }

  //@@ Insert code below to create reference result in CPU
  for (uint i=0; i<NUM_BINS; i++) {
    resultRef[i] = 0;
  }
  for (uint i=0; i<inputLength; i++) {
    resultRef[hostInput[i]]++;
  } 
  for (uint i=0; i<NUM_BINS; i++) {
    if (resultRef[i] > 127) { resultRef[i] = 127; }
  } 

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputSizeB);
  cudaMalloc(&deviceBins, binSizeB);


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputSizeB, cudaMemcpyHostToDevice);


  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, binSizeB);   

  //@@ Initialize the grid and block dimensions here
  uint blockSize = TPB;
  uint gridSize = (inputLength+TPB-1) / TPB;
  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridSize, blockSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  
  //@@ Initialize the second grid and block dimensions here
  blockSize = TPB;
  gridSize = (NUM_BINS+TPB-1) / TPB;

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridSize, blockSize>>>(deviceBins, NUM_BINS);    

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, binSizeB, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  for (uint i=0; i<NUM_BINS; i++) {
    if (resultRef[i] != deviceBins[i]) {
      printf("Different Results! \t host result: %d \t device result %d\n", resultRef[i], deviceBins[i]);
  }

  //@@ Free the GPU memory here


  //@@ Free the CPU memory here


  return 0;
}

