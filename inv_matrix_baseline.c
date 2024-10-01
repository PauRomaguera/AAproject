#include <stdio.h>
#include <stdlib.h>

#define REAL double
#define N 5


static REAL input[N][N];
static REAL inverse[N][N];

int main() 
{
  printf("Matrix size: %dx%d\n", N, N);

  REAL pivot, pivot2, sum;
  int i, j, k;

  // initializes matrix with pseudo-random numbers (always the same)
  srand(0u);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      input[i][j] = ((REAL)rand()/RAND_MAX);

  // initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (i == j)
        inverse[i][j] = 1.0f;
      else
        inverse[i][j] = 0.0f;

  // computes the inverse matrix
  for (i = 0; i < N; i++) 
  {
    pivot = input[i][i];
    for (j = 0; j < N; j++)
    {
      input[i][j] /= pivot;
      inverse[i][j] /= pivot;
    }
    for (j = 0; j < N; j++)
      if (i != j) 
      {
        pivot2 = input[j][i];
        for (k = 0; k < N; k++)
          input[j][k] -= input[i][k]*pivot2;
        for (k = 0; k < N; k++)
          inverse[j][k] -= inverse[i][k]*pivot2;
      }
  }
		
  // computes checksum
  sum = 0.0f;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sum += inverse[i][j];

  printf("Checksum: %.8f\n", sum);
  return 0;
}
