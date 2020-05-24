// #include "parameters.h"
// #include <math.h>
using namespace std;

__global__ void cuda_kernel(int *B, int *A, IndexSave *dInd)
{
	// complete cuda kernel function
	int i = 0;
	int N = 4*LOOP;      											 								// (SIZE)4*(LOOP)4=16
	int TotalThread = gridDim.x * blockDim.x;  								// (BlockNum)4*(ThreadNum)2 = 8
	int stripe = N / TotalThread; 														// stripe = [2,2,2,2,2,2,2,2]
	int head = (blockIdx.x*blockDim.x+threadIdx.x) * stripe; 	//[0,2,4,6,8,10,12,14]
	int LoopLim = head + stripe; 															//[2,4,6,8,10,12,14]
	for(i=head; i<LoopLim; i++)
	{
		B[i] = A[i] * A[i] * A[i] *A[i];
		dInd[i].blockInd_x = blockIdx.x;
		dInd[i].threadInd_x = threadIdx.x;
		dInd[i].head = head;
		dInd[i].stripe = stripe;
	}
};


float GPU_kernel(int *B,int *A,IndexSave* indsave){

	int *dA = 0,*dB;
	IndexSave* dInd;

	// Creat Timing Event
  cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// Allocate Memory Space on Device
	int N = SIZE;																// Size of the memory
	cudaMalloc((void**) &dB, sizeof(int)*N); 		// new a memory size in GPU for array B
	cudaMalloc((void**) &dA, sizeof(int)*N); 		// new a memory size in GPU for array A



	// Allocate Memory Space on Device (for observation)
	cudaMalloc((void**)&dInd, sizeof(IndexSave)*SIZE);

	// Copy Data to be Calculated
	cudaMemcpy(dA, A, sizeof(int)*N, cudaMemcpyHostToDevice); // copy array A to the  GPU
	// Copy Data (indsave array) to device, CPU to GPU
	cudaMemcpy(dInd, indsave, sizeof(IndexSave)*SIZE, cudaMemcpyHostToDevice);

	// Start Timer
	cudaEventRecord(start, 0);

	// Lunch Kernel,method 1
	dim3 dimGrid(2);	// Block Size
	dim3 dimBlock(4); // Thread Size
	cuda_kernel<<<dimGrid,dimBlock>>>(dB, dA, dInd);
	cudaDeviceSynchronize();
	// Stop Timer
	cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

	// Copy Output back
	cudaMemcpy(B, dB, sizeof(int)*N, cudaMemcpyDeviceToHost); // copy array A to the  GPU
	cudaMemcpy(indsave, dInd, sizeof(IndexSave)*SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(A, dA, sizeof(int)*N, cudaMemcpyDeviceToHost); // copy array A to the  GPU
	// Release Memory Space on Device
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dInd);

	// Calculate Elapsed Time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	return elapsedTime;
}
