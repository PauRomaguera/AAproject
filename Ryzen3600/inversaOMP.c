#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define REAL double
#define N 1500

int main() {
  printf("Matrix size: %dx%d\n", N, N);

  REAL *input = (REAL *)aligned_alloc(64, N * N * sizeof(REAL));
  REAL *inverse = (REAL *)aligned_alloc(64, N * N * sizeof(REAL));
  REAL pivot, pivot2, sum;
  int i, j, k;

  // Initializes matrix with pseudo-random numbers (always the same)
  srand(0u);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      input[i * N + j] = ((REAL)rand() / RAND_MAX);

  // Initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      inverse[i * N + j] = (i == j) ? 1.0 : 0.0;

  // Computes the inverse matrix using Gauss-Jordan elimination
  for (i = 0; i < N; i++) {
    pivot = input[i * N + i];

    // Normalize the current row
    for (j = 0; j < N; j++) {
      input[i * N + j] /= pivot;
      inverse[i * N + j] /= pivot;
    }

    // Eliminate the current column in other rows
    #pragma omp parallel for private(j, k, pivot2) shared(input, inverse, i)
    for (j = 0; j < N; j++) {
      if (i != j) {
        pivot2 = input[j * N + i];
        for (k = 0; k < N; k++) {
          input[j * N + k] -= input[i * N + k] * pivot2;
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
        }
      }
    }
  }

  // Computes checksum
  sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sum += inverse[i * N + j];

  free(input);
  free(inverse);
  printf("Checksum: %.8f\n", sum);
  return 0;
}
