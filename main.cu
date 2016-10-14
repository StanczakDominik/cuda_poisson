#include <stdio.h>
#define NX 64
#define NY 64
#define DX (1./(float)NX)
#define DY (1./(float)NY)

// Solves the Poisson equation via the Jacobi Method
// $$\nabla^2 \phi = f$$

__global__ void iteratePoisson(float* d_source, float* d_V1, float* d_V2)
{
    int i = blockIdx.x * blockSize.x + threadIdx.x;
    int j = blockIdx.y * blockSize.y + threadIdx.y;
    int n = j * NX + i;
    if ((i>0) && (i < NX-1) && (j>0) && (j < NY-1)) { //TODO: see what can be done about boundaries
        int n_top = (j-1) * NX + i;
        int n_bot = (j+1) * NX + i;
        int n_left = j * NX + (i-1);
        int n_right = j * NX + (i+1);
        //TODO: rewrite above in terms of n?

        d_V1[n] = 0.25f * (d_V2[n_top] + d_V2[n_bot] + d_V2[n_left] + d_V2[n_right] +\
            d_source[n] * DX * DY;
    }
}

int main()
{
    float *h_source = new float[NX*NY];
    float *h_V = new float[NX*NY];

    float top_bc = 1;
    float bottom_bc = -1;
    float left_bc = 1;
    float right_bc = -1;

    for(int j =0; j<NY; j++){
        for (int i = 0; i < NX; i++){
            int n = NX*j + i;
            if (j == 0){ // top row
                h_V[n] = top_bc;
            }
            else if (j==NY-1){ //bottom row
                h_V[n] = bottom_bc;
            }
            else if (i==0){ //left column
                h_V[n] = left_bc;
            }
            else if (i==NX-1){
                h_V[n] = right_bc;
            }
            printf("%d %d %.3f\n", i, j, h_V[n]);
        }
    }

    float *d_source;
    float *d_V1;
    float *d_V2;
}
