#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>

// Global Variables
#define dim_matrix 8192
#define inf 1000000
#define error 1e-3
#define damping 0.8
#define blockSize 128

__global__ void parMult(double *TM,double *v,double *output){

    int col = blockIdx.x*blockDim.x+threadIdx.x;
    double Pvalue;
    if(col < dim_matrix){
        Pvalue = 0;
        for (int k = 0; k < dim_matrix ; k++){
            Pvalue += v[k] * TM[k*dim_matrix+col];
        }
        output[col] = Pvalue;
    }

}



__global__ void count_ones_par(double *input,int *output){
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	if(col < dim_matrix){	
		double temporal = 0;
		for(int j =0; j < dim_matrix; j++){
			temporal += input[col*dim_matrix+j];
		}
		output[col]=temporal;
	}
}



void count_ones(double *input,int *output,int d1){
	for(int i = 0; i < d1; i++){
		double temporal = 0;
		for(int j =0; j < d1; j++){
			temporal += input[i*d1+j];
		}
		output[i]=temporal;
	}
}

void transitionMatrix(double *input, double *v, double *last_v,int *ones, double *output,int d1){
	for(int i = 0; i < d1; i++){
		v[i] = (1.0/(1.0*d1));
		last_v[i] = inf;
		for(int j = 0; j < d1; j++){	
			if(ones[i]>0) output[i*d1+j]=input[i*d1+j]*(1.0/(1.0*ones[i]));
			else output[i*d1+j]=0.0;
		}
	}
}

void M_hat(double *input, double *output,int d1){
	int size = d1* d1;
	for(int i=0; i < size; i++){
		output[i] = 0.8 * input[i] + ((0.2)/(1.0*d1));
	}
}

void copy_values(double *input,double *output,int dim1){
	for(int i = 0; i < dim1; i++){
		output[i] = input[i];
	}
}

__global__ void transitionMatrix_par(double *input, double *v, double *last_v,int *ones, double *output){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	if((col < dim_matrix) && (row < dim_matrix)){	
		v[col] = (1.0/(1.0*dim_matrix));
		last_v[col] = inf;	
		if(ones[col]>0) output[col*dim_matrix+row]=input[col*dim_matrix+row]*(1.0/(1.0*ones[col]));
		else output[col*dim_matrix+row]=0.0;
	}
}




__global__ void M_hat_par(double *input, double *output){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if((row < dim_matrix) &&  (col < dim_matrix)){
		output[col*dim_matrix+row]=0.8*input[col*dim_matrix+row] + ((0.2))/(1.0*dim_matrix);		
	}
}


void fillMatrix(double *a,int rows,int cols){
	for(int i = 0; i < rows ; i++){
		for(int j = 0; j < cols; j++){
			//if(i == j) a[i*cols+j] = 0;
			//else a[i*cols+j] =1;
			//else {
				int x;
				scanf("%d",&x);
				a[i*cols+j] = x;
			//}
		}
	}
}



__global__ void copy_values_par(double *input, double *output){
	/*int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(col < dim_matrix){
		output[col] = input[col];
	}*/
	for(int i=0; i < dim_matrix ; i++){
		output[i] = input[i];
	}
}





__global__ void quadratic_error(double *v,double *last_v,double *res){
	double resultado = 0.0;
	for(int i = 0; i < dim_matrix; i++){
		//printf("%.15lf %.15lf\n",v[i],last_v[i]);
		resultado += (v[i]-last_v[i])*(v[i]-last_v[i]);
	}
	resultado = sqrt(resultado);
	*res= resultado;
}



void printer(double *mat,int dim){
	int size = dim*dim;
	for(int i = 0; i < size ; i++){
		if(i%dim == 0) printf("\n");
		printf("%lf ",mat[i]);
	}
	printf("\n");
}

void printer2(double *mat,int dim){
	for(int i = 0; i < dim ; i++){
		printf("\n");
		printf("%.15lf ",mat[i]);
	}
	printf("\n");
}





int main(){
	int size1 		= dim_matrix * dim_matrix * sizeof(double);
	int size2		= dim_matrix * sizeof(double);
	double *a		= (double*) malloc(size1);
	double *tm   	= (double*) malloc(size1);
	double *m_hat   = (double*) malloc(size1);
	double *v   	= (double*) malloc(size2);
	double *last_v 	= (double*) malloc(size2);
	double *v_prueba= (double*) malloc(size2);
	double *d_error;
	double h_error;
	int *ones 		= (int*) malloc(size2);



	int blocks      =   ceil(dim_matrix/(1.0*blockSize));

    double *m_hatInDevice;
    double *vInDevice;
    double *last_vInDevice;
    double *aInDevice;
    double *tmInDevice;
    int *onesInDevice;



	clock_t  startGPU, endGPU, sgpu2,egpu2;
    double gpu_time_used,gpu,time_spent=1.1109380000;



    cudaMalloc((void **) &m_hatInDevice, size1);
    cudaMalloc((void **) &vInDevice, size2);
    cudaMalloc((void **) &last_vInDevice, size2);
    cudaMalloc((void **) &onesInDevice, size2);
    cudaMalloc((void **) &aInDevice, size1);
    cudaMalloc((void **) &tmInDevice, size1);

    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(blocks,blocks,1);

	fillMatrix(a,dim_matrix,dim_matrix);

    cudaMemcpy(aInDevice, a, size1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	count_ones_par<<<dimGrid,dimBlock>>>(aInDevice,onesInDevice);
	cudaDeviceSynchronize();
	transitionMatrix_par<<<dimGrid,dimBlock>>>(aInDevice,vInDevice ,last_vInDevice,onesInDevice,tmInDevice);
	cudaDeviceSynchronize();
	M_hat_par<<<dimGrid,dimBlock>>>(tmInDevice,m_hatInDevice);
	cudaDeviceSynchronize();
	int steps = 0;
	cudaMalloc(&d_error, sizeof(double));	
	cudaDeviceSynchronize();
	startGPU = clock();
	//while(true){
		quadratic_error<<<1,1>>>(vInDevice,last_vInDevice,d_error); 

		//cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
		//if(h_error <= error) break;
		cudaMemcpy(last_vInDevice,vInDevice,size2, cudaMemcpyDeviceToDevice);
		parMult<<<dimGrid,dimBlock>>> (m_hatInDevice,last_vInDevice,vInDevice);
		steps++;
	//}
	endGPU = clock();


	cudaMemcpy(v, vInDevice, size2, cudaMemcpyDeviceToHost);

    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    gpu =((double) (egpu2 - sgpu2 )) / CLOCKS_PER_SEC;

    printf("%.10f\n", gpu_time_used);
    //printf("Tiempo algoritmo2 paralelo: %.10f\n", time_spent);

	//printf("converge en: %d\n",steps);
	//printer2(v,dim_matrix);


    cudaFree(m_hatInDevice);
    cudaFree(vInDevice);
    cudaFree(last_vInDevice);
    cudaFree(aInDevice);
    cudaFree(tmInDevice);
    cudaFree(last_vInDevice);
    cudaFree(d_error);
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    free(a);
    free(tm );
    free(m_hat );
	free(v );
	free(last_v ); 
	free(ones ); 
    ////////////////////////////////////////////////////////////////
	return 0;
}


