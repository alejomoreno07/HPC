#include <stdio.h>
#include <stdlib.h>

#define ndim1 10
#define ndim2 10
#define ndim3 10


__global__ void matrixMulKernel(int *d_M, int *d_N, int *d_P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < width)&&(col < width)){
        Pvalue = 0;
        for (int k = 0; k < width ; ++k){
            Pvalue += d_M[row*width+k] * d_N[k*width+col];
        }
        d_P[row*width+col] = Pvalue;
    }
}

void multiplicacionMatrices(double *a,double *b,double *c,int d1,int d2,int d3){
    for(int i = 0; i < d3; i++){
        for(int j = 0; j < d1; j++){
            double temporal=0;
            for(int k = 0; k <d2 ; k++){
                temporal+=b[i*d2+k]*a[k+j*d2];
            }
            c[i*d3+j]=temporal;

        }
    }
}

void fillMatrix(double *a,int rows,int colomns){
    for(int i = 0; i < rows ; i++){
        for(int j = 0; j < colomns ; j++){
            a[i*colomns+j] = i*colomns+j;
        }
    }
}

void printer(double *a,double *b,double *c,int dim1,int dim2,int dim3){
    for(int i= 0; i < dim1*dim2 ;i++) printf("%lf ",a[i]);
    printf("\n");
    for(int i= 0; i < dim2*dim3 ;i++) printf("%lf ",b[i]);
    printf("\n");
    for(int i= 0; i < dim1*dim3 ;i++) printf("%lf ",c[i]);
    printf("\n");
}

 
int main(){

    double *a    =  (double*) malloc(ndim1 * ndim2 * sizeof(double) );
    double *b    =  (double*) malloc(ndim2 * ndim3 * sizeof(double) );
    double *c     =  (double*) malloc(ndim1 * ndim3 * sizeof(double) );
    fillMatrix(a,ndim1,ndim2);
    fillMatrix(b,ndim2,ndim3);
    multiplicacionMatrices(a,b,c,ndim1,ndim2,ndim3);
    printer(a,b,c,ndim2,ndim2,ndim3);
    free(a);
    free(b);
    free(c);
    
    return 0;
}