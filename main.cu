#include <stdio.h>
#define NX 64
#define NY 64
#define DX (1./(float)NX)
#define DY (1./(float)NY)
#define N_ITERATIONS 1000

// Solves the Poisson equation via the Jacobi Method
// $$\nabla^2 \phi = f$$

__global__ void iteratePoisson(float* d_source, float* d_V1, float* d_V2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n = j * NX + i;
    if ((i>0) && (i < NX-1) && (j>0) && (j < NY-1)) { //TODO: see what can be done about boundaries
        int n_top = (j-1) * NX + i;
        int n_bot = (j+1) * NX + i;
        int n_left = j * NX + (i-1);
        int n_right = j * NX + (i+1);
        //TODO: rewrite above in terms of n?

        d_V1[n] = 0.25f * (d_V2[n_top] + d_V2[n_bot] + d_V2[n_left] + d_V2[n_right]) +\
            d_source[n] * DX * DY;
            //TODO: check above for consistency. Does this need factor of 4?
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

    FILE* file_V1 = fopen("V1.dat", "w");
    FILE* file_source = fopen("source.dat", "w");

    float x;
    float y;

    for(int j =0; j<NY; j++){
        for (int i = 0; i < NX; i++){
            int n = NX*j + i;

            x = i*DX;
            y = j*DY;
            h_source[n] = 0.f;  //TODO: set up source term

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

            fprintf(file_V1, "%d %d %.3f %.3f %.3f\n", i, j, x, y, h_V[n]);
            fprintf(file_source, "%d %d %.3f %.3f %.3f\n", i, j, x, y, h_source[n]);
        }
    }


    float *d_source;
    float *d_V1;
    float *d_V2;

    //TODO: allocate GPU memory

    //TODO: copy V1, V2 from host to device

    //TODO: set up blocks
    // blocks =
    // threads =
    // for (int i = 0; i < N_ITERATIONS; i += 2)
    // {
    //     iteratePoisson<<<blocks, threads>>>(d_source, d_V1, d_V2);
    //     cudaDeviceSynchronize();
    //     // cuda get error
    //     iteratePoisson<<<blocks, threds>>>(d_source, d_V2, d_V1);
    //     cudaDeviceSynchronize();
    //     //cuda get error
    // }

    //TODO: copy V2 from device to host as final value

    //TODO: write data out
}
