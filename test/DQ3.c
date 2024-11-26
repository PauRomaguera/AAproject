#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define REAL double
#define N 1500
#define THRESHOLD 32  // Direct inversion threshold for small submatrices

void invertMatrix(REAL *A, REAL *inverse, int n);
void directInvert(REAL *A, REAL *inverse, int n);
void multiplyMatrices(REAL *A, REAL *B, REAL *result, int n);
void subtractMatrices(REAL *A, REAL *B, REAL *result, int n);
void copyMatrix(REAL *src, REAL *dest, int n);
void initializeMatrix(REAL *matrix, int size, REAL value);

int main() 
{
    printf("Matrix size: %dx%d\n", N, N);

    REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
    REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
    REAL sum;
    int i, j;

    // Initialize matrix with pseudo-random numbers (always the same)
    srand(0u);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            input[i*N + j] = ((REAL)rand() / RAND_MAX) + 1e-3;  // Ensure non-zero values

    // Compute the inverse matrix using divide and conquer
    initializeMatrix(inverse, N * N, 0.0);  // Clear inverse matrix before use
    invertMatrix(input, inverse, N);

    // Compute checksum
    sum = 0.0;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            sum += inverse[i * N + j];
    free(input);
    free(inverse);

    printf("Checksum: %.8f\n", sum);
    return 0;
}

void invertMatrix(REAL *A, REAL *inverse, int n) {
    if (n <= THRESHOLD) {
        directInvert(A, inverse, n);  // Direct inversion for small matrices
        return;
    }

    int half = n / 2;
    REAL *A11 = A;
    REAL *A12 = A + half;
    REAL *A21 = A + n * half;
    REAL *A22 = A + n * half + half;

    REAL *A11_inv = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *A22_inv = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *S = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *temp = (REAL *)malloc(half * half * sizeof(REAL));

    initializeMatrix(A11_inv, half * half, 0.0);
    initializeMatrix(A22_inv, half * half, 0.0);
    initializeMatrix(S, half * half, 0.0);
    initializeMatrix(temp, half * half, 0.0);

    // Recursive inversion of A11 and A22
    invertMatrix(A11, A11_inv, half);
    invertMatrix(A22, A22_inv, half);

    // Compute Schur complement: S = A22 - A21 * A11_inv * A12
    REAL *A21_A11_inv = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *A21_A11_inv_A12 = (REAL *)malloc(half * half * sizeof(REAL));
    initializeMatrix(A21_A11_inv, half * half, 0.0);
    initializeMatrix(A21_A11_inv_A12, half * half, 0.0);
    multiplyMatrices(A21, A11_inv, A21_A11_inv, half);
    multiplyMatrices(A21_A11_inv, A12, A21_A11_inv_A12, half);
    subtractMatrices(A22, A21_A11_inv_A12, S, half);

    // Invert S (Schur complement)
    REAL *S_inv = (REAL *)malloc(half * half * sizeof(REAL));
    initializeMatrix(S_inv, half * half, 0.0);
    invertMatrix(S, S_inv, half);

    // Calculate blocks of the inverse matrix
    REAL *B11 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B12 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B21 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B22 = (REAL *)malloc(half * half * sizeof(REAL));

    initializeMatrix(B11,

