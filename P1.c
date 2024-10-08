#include <stdio.h>
#include <stdlib.h>

#define REAL double
#define N 1500

int main() 
{
  printf("Matrix size: %dx%d\n", N, N);

  REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
  REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
  REAL pivot, pivot2, sum;
  int i, j, k;

  // initializes matrix with pseudo-random numbers (always the same)
  /*
  srand(0u);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      input[i*N+j] = ((REAL)rand()/RAND_MAX);
*/
    srand(0u);
    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            input[i * N + j] = ((REAL)rand() / RAND_MAX);
            input[j * N + i] = input[i * N + j]; // Make the matrix symmetric
        }
        input[i * N + i] += N; // Ensure positive definiteness
    }

  // initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (i == j)
        inverse[i*N+j] = 1.0;
      else
        inverse[i*N+j] = 0.0;

  // computes the inverse matrix
  for (i = 0; i < N; i++) 
  {
    pivot = input[i*N+i];
    for (j = 0; j < N; j++)
    {
      input[i*N+j] /= pivot;
      inverse[i*N+j] /= pivot;
    }
    for (j = 0; j < N; j++)
      if (i != j) 
      {
        pivot2 = input[j*N+i];
        for (k = 0; k < N; k++) {
          input[j * N + k] -= input[i * N + k] * pivot2;
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
	  }
       // for (k = 0; k < N; k++)
        //  inverse[j * N + k] -= inverse[i * N + k] * pivot2;
      }
  }
		
  // computes checksum
  sum = 0.0f;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sum += inverse[i * N + j];

  printf("Checksum: %.8f\n", sum);
  return 0;
}
