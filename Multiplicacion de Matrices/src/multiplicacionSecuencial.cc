#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <fstream>

//////////////////////////////////////
//      GLOBAL VARIABLES           //
/////////////////////////////////////
#define ndim1 32
#define ndim2 32
#define ndim3 32
/////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////||
//                          MULTIPLICACION SECUENCIAL                               ||
////////////////////////////////////////////////////////////////////////////////////||

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

/////////////////////////////////////////////////////////////////////////////////////

void fillMatrix(double *a,int rows,int colomns){
    for(int i = 0; i < rows ; i++){
        for(int j = 0; j < colomns ; j++){
            a[i*colomns+j] = i*colomns+j+1;
        }
    }
}

void printer(double *a,double *b,double *c,int dim1,int dim2,int dim3){
    /*for(int i= 0; i < dim1*dim2 ;i++) printf("%lf ",a[i]);
    printf("\n");
    for(int i= 0; i < dim2*dim3 ;i++) printf("%lf ",b[i]);
    printf("\n");*/
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
    

    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    //          time measuring VARIABLES                                //
    /////////////////////////////////////////////////////////////////////
    clock_t start, end;
    double cpu_time_used;
    ////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    fillMatrix(a,ndim1,ndim2);
    fillMatrix(b,ndim2,ndim3);
    /////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////
    start = clock();
    multiplicacionMatrices(a,b,d,ndim1,ndim2,ndim3);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo algoritmo secuencial: %.10f\n", cpu_time_used);
    ////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////
    free(a);
    free(b);
    free(c);
    free(d);
    ////////////////////////////////////////////////////////////////
    
    return 0;
}