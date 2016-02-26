#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
 
#define lim 1000000
#define blockSize 1024

__global__ 
void summ(int *a,int *b,int *c){
  int i=threadIdx.x;
	c[i]= a[i]+b[i];
}

__global__ 
void sumatoria(int *a,int *b,int *c){
  int i=blockIdx.x;
	c[i]= a[i]+b[i];
}

__global__ 
void sumaFinal(int *a,int *b,int *c){
     int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i > lim-1) return;
	c[i]= a[i]+b[i];
}

__global__
void sumat(int *a,int *b,int *c){
	for(int i = 0; i < lim; i++){
		c[i]= a[i]+b[i];
	}
}
 
void printer(int *a, int *b, int *c){
	for(int i = lim-10 ; i < lim ; i++){  printf("%d ",a[i]); }
	printf("\n");
	for(int i = lim-10 ; i < lim ; i++){  printf("%d ",b[i]); }
	printf("\n");
	for(int i = lim-10 ; i < lim ; i++){  printf("%d ",c[i]); }
	printf("\n");
}
 
int main(){

	int *a,*b;
	int *suma;
	size_t size=lim*sizeof(int);
  	a=(int*)malloc(size);
	b=(int*)malloc(size);
  	suma=(int*)malloc(size);
	for(int i = 0; i < lim ; i++){
		a[i]=1;
		b[i]=1;
	}

	int *aInDevice, *bInDevice, *sumaInDevice;
  clock_t begin, end;
	double time_spent;
	begin = clock();
	cudaMalloc((void **) &aInDevice, size ); 
	cudaMalloc((void **) &bInDevice, size );
	cudaMalloc((void **) &sumaInDevice, size );  
	cudaMemcpy(aInDevice, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bInDevice, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(sumaInDevice, suma, size, cudaMemcpyHostToDevice);
	//summ <<< 1,10 >>> (aInDevice,bInDevice,sumaInDevice);
  //sumat <<< 1,10 >>> (aInDevice,bInDevice,sumaInDevice);
	int blocks = ceil(lim/(1.0*blockSize));
	sumaFinal <<< blocks, blockSize >>> (aInDevice,bInDevice,sumaInDevice);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("%lf\n",time_spent);
    
	cudaMemcpy(suma, sumaInDevice, size, cudaMemcpyDeviceToHost);
	printer(a,b,suma);
  cudaFree(aInDevice);
	cudaFree(bInDevice);
	cudaFree(sumaInDevice);
	
	free(a);
	free(b);
	free(suma);
	
	return 0;
}