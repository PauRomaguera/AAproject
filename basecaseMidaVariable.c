#include <stdio.h>
#include <stdlib.h>

#define REAL double


int main(int argc, char *argv[])
{
  if (argc != 2) {
      printf("Usage: %s <matrix_size>\n", argv[0]);
      return 1;
  }
  int N = atoi(argv[1]);
  printf("Matrix size: %dx%d\n", N, N);

  REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
  REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
  REAL pivot, pivot2, sum;
  int i, j, k;

  // initializes matrix with pseudo-random numbers (always the same)
//  double A[16] = {3, 1, 4, 2,
 //                   12, 37, -43, 0,
   //                 -16, -43, 98, 0,
     //               0, 0, 0, 1};
  srand(0u);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      input[i*N+j] = ((REAL)rand()/RAND_MAX);

  // initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (i == j)
        inverse[i*N+j] = 1.0f;
      else
        inverse[i*N+j] = 0.0f;

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
        for (k = 0; k < N; k++)
          input[j * N + k] -= input[i * N + k] * pivot2;
        for (k = 0; k < N; k++)
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
      }
  }
		
  // computes checksum
  sum = 0.0f;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sum += inverse[i * N + j];
/*  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
        printf("%8.4f ", inverse[i * N + j]);
    }
    printf("\n");
  }*/
  printf("\n");
  free(input);
  free(inverse);
  printf("Checksum: %.8f\n", sum);
  return 0;
}
