#include <stdio.h>
#include <stdlib.h>

//////////////////////////////////////
//      GLOBAL VARIABLES           //
/////////////////////////////////////
#define ndim1 3
#define ndim2 3
#define ndim3 3
#define blockSize 32
/////////////////////////////////////


__global__ void matrixMulKernel(double *d_M, double *d_N, double *d_P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < width)&&(col < width)){
        Pvalue = 0;
        for (int k = 0; k < width ; ++k){
            //printf("%lf %lf %lf\n",d_M[row*width+k],d_N[k*width+col],d_M[row*width+k]*d_N[k*width+col]);
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
                temporal+=b[i*d2+k]*a[k*d2+j];
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
    //for(int i= 0; i < dim1*dim2 ;i++) printf("%lf ",a[i]);
    //printf("\n");
    //for(int i= 0; i < dim2*dim3 ;i++) printf("%lf ",b[i]);
    //printf("\n");
    for(int i= 0; i < dim2*dim3 ;i++) printf("%lf ",c[i]);
    printf("\n");
}

 
int main(){
    /////////////////////////////////////////////////////////////////////
    int size1       =   ndim1 * ndim2 * sizeof(double);
    int size2       =   ndim2 * ndim3 * sizeof(double);
    int size3       =   ndim1 * ndim3 * sizeof(double);
    double *a       =   (double*) malloc(size1);
    double *b       =   (double*) malloc(size2);
    double *c       =   (double*) malloc(size3);
    double *d       =   (double*) malloc(size3);
    
    int blocks      =   ceil(ndim1/(1.0*blockSize));
    double *aInDevice;
    double *bInDevice; 
    double *cInDevice;
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    fillMatrix(a,ndim1,ndim2);
    fillMatrix(b,ndim2,ndim3);
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    cudaMalloc((void **) &aInDevice, size1); 
    cudaMalloc((void **) &bInDevice, size2);
    cudaMalloc((void **) &cInDevice, size3);  
    cudaMemcpy(aInDevice, a, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(bInDevice, b, size2, cudaMemcpyHostToDevice);
    ////////////////////////////////////////////////////////////////////
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(blocks,blocks,1);
    ////////////////////////////////////////////////////////////////////
    matrixMulKernel<<<dimGrid,dimBlock>>> (aInDevice,bInDevice,cInDevice,ndim1);
    ///////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    cudaMemcpy(c, cInDevice, size3, cudaMemcpyDeviceToHost);
    //////////////////////////////////////////////////////////////////
    multiplicacionMatrices(a,b,d,ndim1,ndim2,ndim3);
    /////////////////////////////////////////////////////////////////
    printer(a,b,c,ndim2,ndim2,ndim3);
        printer(a,b,d,ndim2,ndim2,ndim3);
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    cudaFree(aInDevice);
    cudaFree(bInDevice);
    cudaFree(cInDevice);
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    free(a);
    free(b);
    free(c);
    free(d);
    ////////////////////////////////////////////////////////////////
    
    return 0;
}