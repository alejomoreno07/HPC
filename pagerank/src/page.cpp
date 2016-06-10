#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <math.h>

// Global Variables
#define dim_matrix 8192
#define inf 1000000
#define error 1e-3
#define damping 0.8

void lastMult(double *TM,double *v,double *output, int d1){
	for(int j = 0; j < d1 ; j++){
		double temporal = 0;
		for(int k = 0; k < d1; k++){
			temporal += v[k]*TM[k*d1+j];
		}
		output[j]=temporal;
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

void fillMatrix(double *a,int rows,int cols){
	srand(time(NULL));
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

void copy_values(double *input,double *output,int dim1){
	for(int i = 0; i < dim1; i++){
		output[i] = input[i];
	}
}

double quadratic_error(double *v,double *last_v,int dim1){
	double resultado = 0.0;
	for(int i = 0; i < dim1; i++){
		resultado += (v[i]-last_v[i])*(v[i]-last_v[i]);
	}
	resultado = sqrt(resultado);
	return resultado;
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
	double *b 		= (double*) malloc(size2);
	double *tm   	= (double*) malloc(size1);
	double *m_hat   = (double*) malloc(size1);
	double *v   	= (double*) malloc(size2);
	double *last_v 	= (double*) malloc(size2);
	int *ones 		= (int*) malloc(size2);
	 clock_t start, end, s2, e2;
    double cpu_time_used , cpu;
	fillMatrix(a,dim_matrix,dim_matrix);
	count_ones(a,ones,dim_matrix);
	transitionMatrix(a,v,last_v,ones,tm,dim_matrix);
	M_hat(tm,m_hat,dim_matrix);
	int steps = 0;
		start = clock();
	//while(quadratic_error(v,last_v,dim_matrix) > error){
		quadratic_error(v,last_v,dim_matrix);
		copy_values(v,last_v,dim_matrix);
		lastMult(m_hat,last_v,v,dim_matrix);
		steps++;
	//}
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cpu = ((double) (e2 - s2)) / CLOCKS_PER_SEC;
    printf("%.10f\n", cpu_time_used);
	return 0;
}


