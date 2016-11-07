#include <stdio.h>
#define NX 8
#define NY 8
#define DX (1./(float)NX)
#define DY (1./(float)NY)
#define N_ITERATIONS 8
#define N_THREADS 512
#define N_BLOCKS (NX*NY+N_THREADS+1)/N_THREADS


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
    float *h_source = (float *)malloc(NX*NY*sizeof(float));
    float *h_V = (float *)malloc(NX*NY*sizeof(float));

    float top_bc = 1;
    float bottom_bc = -1;
    float left_bc = 1;
    float right_bc = -1;

    float x;
    float y;

    for(int j =0; j<NY; j++){
        for (int i = 0; i < NX; i++){
            int n = NX*j + i;

            x = i*DX - NX/2 * DX;
            y = j*DY - NY/2 * DY;
            h_source[n] = x*x+y*y;  //TODO: set up source term

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
        }
    }


    float *d_source;
    float *d_V1;
    float *d_V2;

    //allocate GPU memory
    gpuErrchk(cudaMalloc(&d_source, NX*NY*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_V1, NX*NY*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_V2, NX*NY*sizeof(float)));

    // copy V1, V2 from host to device
    gpuErrchk(cudaMemcpy(d_source, h_source, NX*NY*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_V1, h_V, NX*NY*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_V2, h_V, NX*NY*sizeof(float), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaPeekAtLastError());

    printf("Blocks: %d\nThreads: %d\n", N_BLOCKS, N_THREADS);
    printf("Iteration %5d", 0);
    for (int i = 0; i < N_ITERATIONS; i += 2)
    {
        printf("\rIteration %5d", i);
        iteratePoisson<<<N_BLOCKS, N_THREADS>>>(d_source, d_V1, d_V2);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        iteratePoisson<<<N_BLOCKS, N_THREADS>>>(d_source, d_V2, d_V1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    //copy V2 from device to host as final value
    gpuErrchk(cudaMemcpy(h_source, d_source, NX*NY*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V2, NX*NY*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_source));
    gpuErrchk(cudaFree(d_V1));
    gpuErrchk(cudaFree(d_V2));

    FILE* file_V1 = fopen("V1.dat", "w");
    FILE* file_source = fopen("source.dat", "w");

    for(int j =0; j<NY; j++){
        for (int i = 0; i < NX; i++){
            int n = NX*j + i;

            x = i*DX;
            y = j*DY;

            fprintf(file_V1, "%d %d %.3f %.3f %.3f\n", i, j, x, y, h_V[n]);
            fprintf(file_source, "%d %d %.3f %.3f %.3f\n", i, j, x, y, h_source[n]);
        }
    }
    //free GPU arrays
    free(h_V);
    free(h_source);

    //write data out
    printf("\nFinished!\n");
}
