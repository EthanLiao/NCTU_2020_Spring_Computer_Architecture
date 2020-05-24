#include <stdio.h>
#include "parameters.h"
#include "device_1.cu"
#include <cuda.h>
extern float GPU_kernel(int *B, int *A, IndexSave* indsave);

void genNumbers(int *number, int size){
	for(int i = 0; i < size; i++)
		number[i] = rand()%256;
}

// this function is the core function for calculating
void function_1(int B[],int A[]){
	for(int i=0;i<SIZE;i++){
		B[i]=A[i];

		for(int j=1;j<LOOP;j++)
		B[i]*=A[i];

	}
}


bool verify(int A[],int B[]){

	for(int i=0;i<SIZE;i++){
		if(A[i]!=B[i]) return true;
	}
	return false;
}

void printIndex(IndexSave* indsave,int *B,int *C)
{
	for(int i=0;i<SIZE;i++)
	{
		printf("%d : blockInd_x=%d,threadInd_x=%d,head=%d,stripe=%d",i,(indsave[i]).blockInd_x,(indsave[i]).threadInd_x,(indsave[i]).head,(indsave[i]).stripe);
		printf(" || GPU result=%d,CPU result=%d\n",B[i],C[i]);
	}
}

int main()
{
	// random seed
	int *A=new int[SIZE];
	// random number sequence computed by GPU
	int *B=new int[SIZE];
	// random number sequence computed by CPU
	int *C=new int[SIZE];
	// Indices saver (for checking correctness)
	IndexSave *indsave = new IndexSave[SIZE];

	genNumbers(A,SIZE);
	memset( B, 0, sizeof(int)*SIZE );
	// for(int i=0;i<SIZE;i++){
	// 	printf("%d \n",A[i]);
	// }
	/* CPU side*/
	function_1(C,A);

	// /* GPU side*/
	float elapsedTime = GPU_kernel(B,A,indsave);

	/*Show threads execution info*/
	printIndex(indsave,B,C);

	printf("==============================================\n");
	/* verify the result*/
	if(verify(B,C)){printf("wrong answer\n");}
	printf("GPU time = %5.2f ms\n", elapsedTime);

	/*Please press any key to exit the program*/
	getchar();

}
