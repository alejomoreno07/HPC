#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <math.h>

// Global Variables
#define dim_matrix  8192
#define inf 1000000
#define error 1e-3
#define damping 0.8



void fillMatrix(double *a,int rows,int cols){
	srand(time(NULL));
	for(int i = 0; i < rows ; i++){
		for(int j = 0; j < cols; j++){
			if(i == j) a[i*cols+j] = 0;
			else a[i*cols+j] =(int) rand()%2;
			//else {
				//int x;
				//scanf("%d",&x);
				//a[i*cols+j] = x;
			//}
		}
	}
}



int main(){
	int size1 		= dim_matrix * dim_matrix * sizeof(double);
	int size2		= dim_matrix * sizeof(double);
	double *a		= (double*) malloc(size1);
	fillMatrix(a,dim_matrix,dim_matrix);	
	for(int i=0; i < dim_matrix; i++){
    	for(int j=0; j < dim_matrix; j++){
    		printf("%d ",(int)a[i*dim_matrix+j]);
    	}
    	printf("\n");
    }
    return 0;
}
