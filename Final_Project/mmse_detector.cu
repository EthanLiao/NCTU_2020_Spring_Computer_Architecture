#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
#define BLOCK_SIZE 16

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define random(x) (rand()%x)

/* GPU matrix add*/
__global__ void gpu_add_matrix(double* matrixA, double* matrixB, double* matrixC,unsigned int row,unsigned int col)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;

	if (i < row && j < col)
	{
		matrixC[i*row+j] = matrixA[i*row+j] + matrixB[i*row+j];
	}
}

__global__ void gpu_matrix_mult(double *a,double *b, double *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_square_matrix_mult(double *left, double *right, double *res, int dim) {
    int i,j;
    float temp = 0;
    __shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];
    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {
        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem
        Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
        // Load right[i][j] to shared mem
        Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        // Synchronize before computation
        __syncthreads();
        // Accumulate one tile of res from tiles of left and right in shared mem
        for (int k = 0; k < BLOCK_SIZE; k++) {
            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col] = temp;
}

__global__ void gpu_matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

void cpu_matrix_mult(double *h_a, double *h_b, double *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*CUDA inverse*/
#define random(x) (rand()%x)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


void printMatrix(double* inputMatrix, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << inputMatrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

double* modifyMatrix(double* inputMatrix, const int rows, const int cols)
{
	double *h_arr;
	double *host_mat = (double*)malloc(rows*cols*sizeof(double));
	cudaMallocHost((void **) &h_arr, sizeof(double)*rows*cols);
	cudaMemcpy(h_arr, inputMatrix, sizeof(double)*rows*cols, cudaMemcpyDeviceToHost);
  return h_arr;
}


/**
 * CUDA kernel that computes reciprocal values for a given vector
 */

__global__ void harnessZeroKernel(double *d_augmentedMatrix, const int rowId1, const int rowId2, const int size) {
	__shared__ double blockR1[512];
	__shared__ double blockR2[512];
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR1[tIdx] = d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx];
		blockR2[tIdx] = d_augmentedMatrix[rowId2 * 2 * size + blockDim.x * bIdx + tIdx];
		__syncthreads();
		d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx] = blockR1[tIdx] + blockR2[tIdx];
	}
}

__global__ void computeRowsKernel(double *d_augmentedMatrix, const int rowId, const int size) {
	__shared__ double blockR[512];
	__shared__ double Aii;
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR[tIdx] = d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx];
		Aii = d_augmentedMatrix[rowId * 2 * size + rowId];
		__syncthreads();
		blockR[tIdx] = blockR[tIdx] / Aii;
		d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx] = blockR[tIdx];
	}
}

__global__ void computeColsKernel(double *d_augmentedMatrix, const int colId, const int size) {
	__shared__ double blockC[16][16];    // which col need to be zero
	__shared__ double blockCCurent[16][16];   // which col is the current col
	__shared__ double ARow[16];        // the pivot row
	const int tIdx = threadIdx.x;
	const int tIdy = threadIdx.y;
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2 && rowI < size) {
		blockC[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colId];
		if (blockC[tIdy][tIdx] != 0) {
			blockCCurent[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colI];
			ARow[tIdx] = d_augmentedMatrix[colId * size * 2 + colI];
			__syncthreads();
			if (rowI != colId) {   // current row can't sub by current row
				blockCCurent[tIdy][tIdx] = blockCCurent[tIdy][tIdx] - blockC[tIdy][tIdx] * ARow[tIdx];
			}
			d_augmentedMatrix[rowI * size * 2 + colI] = blockCCurent[tIdy][tIdx];
			//d_augmentedMatrix[rowI * size * 2 + colI] = ARow[tIdx];
		}
	}
}

__global__ void augmentMatrixKernel(double *d_augmentedMatrix, double *d_inputMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols && rowI < rows) {
			// initialize augmentedMatrix
			if (colI < cols / 2) {
				d_augmentedMatrix[rowI * cols + colI] = d_inputMatrix[rowI * cols / 2 + colI];
			}
			else if (colI - cols / 2 == rowI) {
				d_augmentedMatrix[rowI * cols + colI] = 1;
			} else {
				d_augmentedMatrix[rowI * cols + colI] = 0;
			}

	}
}

__global__ void getInverseMatrixKernel(double *d_augmentedMatrix, double *d_inverseMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols / 2 && rowI < rows) {
			// initialize augmentedMatrix
			d_inverseMatrix[rowI * cols / 2 + colI] = d_augmentedMatrix[rowI * cols + colI + cols / 2];
	}
}

double *gpuMatrixInverse(double *inputMatrix, const int rows, const int cols)
{
	double *h_inverseMatrix;
	double *h_augmentedMatrix;
	double *d_inputMatrix;
	double *d_inverseMatrix;
	double *d_augmentedMatrix;
	const int length = rows * cols;
	const int size = rows;
	// initialization
	h_inverseMatrix = (double *)malloc(length * sizeof(double));
	h_augmentedMatrix = (double *)malloc(length * 2 * sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_augmentedMatrix, sizeof(double) * length * 2));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inputMatrix, sizeof(double) * length));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inverseMatrix, sizeof(double) * length));
	CUDA_CHECK_RETURN(cudaMemcpy(d_inputMatrix, inputMatrix, sizeof(double) * length, cudaMemcpyHostToDevice));

	dim3 blockSize1(16, 16);
	dim3 gridSize1(cols * 2.0 / blockSize1.x + 1, rows * 1.0 / blockSize1.y + 1);
	augmentMatrixKernel<<<gridSize1, blockSize1>>>(d_augmentedMatrix, d_inputMatrix, rows, cols * 2);
	cudaDeviceSynchronize();

	int i = 0;
	while (i < size) {
		if (inputMatrix[i * size + i] != 0) {
			dim3 blockSize2(256);
			dim3 gridSize2(cols * 2.0 / blockSize2.x + 1, 1);
			computeRowsKernel<<<gridSize2, blockSize2>>>(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		} else {
			int nonZeroRowIndex = 0;
			for (int j = 0; j < size; j++) {
				if (inputMatrix[j * size + i] != 0) {
					nonZeroRowIndex = j;
					break;
				}
			}
			dim3 blockSize3(256);
			dim3 gridSize3(cols * 2.0 / blockSize3.x + 1, 1);
			harnessZeroKernel<<<gridSize3, blockSize3>>>(d_augmentedMatrix, i, nonZeroRowIndex, size);
			cudaDeviceSynchronize();
			dim3 blockSize4(256);
			dim3 gridSize4(cols * 2.0 / blockSize4.x + 1, 1);
			computeRowsKernel<<<gridSize4, blockSize4>>>(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		}

		dim3 blockSize5(16, 16);
		dim3 gridSize5(cols * 2.0 / blockSize5.x + 1, rows * 1.0 / blockSize5.y + 1);
		computeColsKernel<<<gridSize5, blockSize5>>>(d_augmentedMatrix, i, size);
		cudaDeviceSynchronize();
		i++;
	}

	dim3 blockSize6(16, 16);
	dim3 gridSize6(cols * 2.0 / blockSize6.x + 1, rows * 1.0 / blockSize6.y + 1);
	getInverseMatrixKernel<<<gridSize1, blockSize1>>>(d_augmentedMatrix, d_inverseMatrix, rows, cols * 2);

	CUDA_CHECK_RETURN(cudaMemcpy(h_inverseMatrix, d_inverseMatrix, sizeof(double) * length, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_augmentedMatrix, d_augmentedMatrix, sizeof(double) * length * 2, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_augmentedMatrix));
	// CUDA_CHECK_RETURN(cudaFree(d_inverseMatrix));
	CUDA_CHECK_RETURN(cudaFree(d_inputMatrix));
	// return h_inverseMatrix;
  return d_inverseMatrix;
}


int main(int argc, char const *argv[])
{
    int m=2, n=2, k=9;
    double sigm = 10;
    double sig_vec[n][k];

    // read the signal from file
    ifstream file("trans_sig.txt");
    double ch;
		int rcnt=0, colcnt=0;
    while(!file.eof()){
      file>>ch;
      sig_vec[rcnt][colcnt]=ch;
      colcnt++;
      if (colcnt==k)
      {
        colcnt = 0;
        rcnt ++;
      }
    }
    file.close();


    double channel_mat[m][n]={
      {5,6,},
      {7,8,}
    };
    double ident_mat[n][n]={
      {1/sigm,0,},
      {0,1/sigm,}
    };

    double *h_sig, *h_channel, *h_Htrans, *h_Htrans_MUL_H;
    double *h_sigm, *h_psudo_inv, *h_W, *h_rcv;
    cudaMallocHost((void **) &h_sig, sizeof(double)*n*k);
    cudaMallocHost((void **) &h_channel, sizeof(double)*m*n);
    cudaMallocHost((void **) &h_Htrans, sizeof(double)*n*m);
    cudaMallocHost((void **) &h_Htrans_MUL_H, sizeof(double)*n*n);
    cudaMallocHost((void **) &h_sigm, sizeof(double)*n*n);
    cudaMallocHost((void **) &h_psudo_inv, sizeof(double)*n*n);
    cudaMallocHost((void **) &h_W, sizeof(double)*n*m);
    cudaMallocHost((void **) &h_rcv, sizeof(double)*n*k);


    // cudaMallocHost((void **) &h_psudo_inv, sizeof(double)*n*n);
    for(int i=0;i<n;i++)
      for(int j=0;j<k;j++)
        h_sig[i*k+j] = sig_vec[i][j];

    for(int i=0;i<m;i++)
      for(int j=0;j<n;j++)
        h_channel[i*n+j] = channel_mat[i][j];

    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
        h_sigm[i*n+j] = ident_mat[i][j];

    // allocate memory in host RAM
    double *h_c;
    cudaMallocHost((void **) &h_c, sizeof(double)*m*k);
    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    // Allocate memory space on the device
    double  *d_c, *d_sig, *d_channel, *d_Htrans, *d_Htrans_MUL_H, *d_sigm, *d_psudo_inv, *d_W, *d_rcv;
    cudaMalloc((void **) &d_c, sizeof(double)*m*k);
    cudaMalloc((void **) &d_channel, sizeof(double)*m*n);
    cudaMalloc((void **) &d_sig, sizeof(double)*n*k);
    cudaMalloc((void **) &d_Htrans, sizeof(double)*m*n);
    cudaMalloc((void **) &d_Htrans_MUL_H, sizeof(double)*n*n);
    cudaMalloc((void **) &d_sigm, sizeof(double)*n*n);
    cudaMalloc((void **) &d_psudo_inv, sizeof(double)*n*n);
    cudaMalloc((void **) &d_W, sizeof(double)*n*m);
    cudaMalloc((void **) &d_rcv, sizeof(double)*n*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_sig, h_sig, sizeof(double)*n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel, h_channel, sizeof(double)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigm, h_sigm, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_channel, d_sig, d_c, m, n, k);
		/*MMSE Dectector*/
    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_channel, d_Htrans, m, n);
    //add matrix
		gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_Htrans, d_channel, d_Htrans_MUL_H, n);
    gpu_add_matrix<<<dimGrid, dimBlock>>>(d_Htrans_MUL_H, d_sigm, d_psudo_inv, n, n);
    //inverse
    double *inverseMatrixGPU, *tmp_mat;
    tmp_mat = modifyMatrix(d_psudo_inv,n,n);
    inverseMatrixGPU = gpuMatrixInverse(tmp_mat, n, n);
    //mul
		// dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
		// dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
		gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(inverseMatrixGPU, d_Htrans, d_W, n);
		gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_W, d_c, d_rcv, n, m, k);
		// double *h_tmp;
		// cudaMallocHost((void **) &h_tmp, sizeof(double)*m*k);
		// cudaMemcpy(h_tmp, d_rcv, sizeof(double)*n*k, cudaMemcpyDeviceToHost);
		// printMatrix(h_tmp,m,k);


    // Transefr results from device to host
    cudaMemcpy(h_rcv, d_rcv, sizeof(double)*n*k, cudaMemcpyDeviceToHost);
    printMatrix(h_rcv ,n,k);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on GPU: %f ms.\n\n",gpu_elapsed_time_ms);

		ofstream wrtfile;
		wrtfile.open("mmse_rcv_sig");
		for (int i = 0; i < n; i++) {
				for (int j = 0; j < k; j++) {
					if(h_rcv[i * k + j]>=0)
						wrtfile << 1 << "\t";
					else
						wrtfile << -1 << "\t";
				}
				wrtfile << "\n";
		}
		wrtfile<<"Time elapsed on GPU : "<<gpu_elapsed_time_ms <<"ms";
		wrtfile.close();
    // free memory
    cudaFree(d_sig);
    cudaFree(d_channel);
    cudaFree(d_c);
    cudaFree(d_Htrans);
    cudaFree(d_Htrans_MUL_H);
    cudaFree(d_sigm);
    cudaFree(d_psudo_inv);
    cudaFree(d_W);
    cudaFree(d_rcv);
    cudaFreeHost(h_sig);
    cudaFreeHost(h_channel);
    cudaFreeHost(h_c);
    cudaFreeHost(h_Htrans);
    cudaFreeHost(h_Htrans_MUL_H);
    cudaFreeHost(h_sigm);
    cudaFreeHost(h_psudo_inv);
    cudaFreeHost(h_W);
    cudaFreeHost(h_rcv);
    // cudaFreeHost(h_tmp);
    return 0;
}
