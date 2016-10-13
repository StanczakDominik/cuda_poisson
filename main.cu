#include <stdio.h>
#define NX 64
#define NY 64
#define DX (1./(float)NX)
#define DY (1./(float)NY)

int main()
{
    float *h_rho = new float[NX*NY];
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




}
