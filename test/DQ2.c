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

    // Recursive inversion of A11 and A22
    invertMatrix(A11, A11_inv, half);
    invertMatrix(A22, A22_inv, half);

    // Compute Schur complement: S = A22 - A21 * A11_inv * A12
    REAL *A21_A11_inv = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *A21_A11_inv_A12 = (REAL *)malloc(half * half * sizeof(REAL));
    multiplyMatrices(A21, A11_inv, A21_A11_inv, half);
    multiplyMatrices(A21_A11_inv, A12, A21_A11_inv_A12, half);
    subtractMatrices(A22, A21_A11_inv_A12, S, half);

    // Invert S (Schur complement)
    REAL *S_inv = (REAL *)malloc(half * half * sizeof(REAL));
    invertMatrix(S, S_inv, half);

    // Calculate blocks of the inverse matrix
    REAL *B11 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B12 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B21 = (REAL *)malloc(half * half * sizeof(REAL));
    REAL *B22 = (REAL *)malloc(half * half * sizeof(REAL));

    // B11 = A11_inv + A11_inv * A12 * S_inv * A21 * A11_inv
    multiplyMatrices(A12, S_inv, temp, half);
    multiplyMatrices(temp, A21, temp, half);
    multiplyMatrices(A11_inv, temp, B11, half);
    for (int i = 0; i < half * half; i++) {
        B11[i] += A11_inv[i];
    }

    // B12 = -A11_inv * A12 * S_inv
    multiplyMatrices(A11_inv, A12, temp, half);
    multiplyMatrices(temp, S_inv, B12, half);
    for (int i = 0; i < half * half; i++) {
        B12[i] = -B12[i];
    }

    // B21 = -S_inv * A21 * A11_inv
    multiplyMatrices(S_inv, A21, temp, half);
    multiplyMatrices(temp, A11_inv, B21, half);
    for (int i = 0; i < half * half; i++) {
        B21[i] = -B21[i];
    }

    // B22 = S_inv
    copyMatrix(S_inv, B22, half);

    // Combine blocks into the result matrix
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            inverse[i * n + j] = B11[i * half + j];
            inverse[i * n + (j + half)] = B12[i * half + j];
            inverse[(i + half) * n + j] = B21[i * half + j];
            inverse[(i + half) * n + (j + half)] = B22[i * half + j];
        }
    }

    // Free temporary storage
    free(A11_inv);
    free(A22_inv);
    free(S);
    free(temp);
    free(S_inv);
    free(A21_A11_inv);
    free(A21_A11_inv_A12);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
}

void directInvert(REAL *A, REAL *inverse, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                inverse[i * n + j] = 1.0;
            else
                inverse[i * n + j] = 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        REAL pivot = A[i * n + i];
        if (fabs(pivot) < 1e-12) {
            printf("Matrix is singular or nearly singular.\n");
            exit(1);
        }
        for (int j = 0; j < n; j++) {
            A[i * n + j] /= pivot;
            inverse[i * n + j] /= pivot;
        }
        for (int j = 0; j < n; j++) {
            if (i != j) {
                REAL factor = A[j * n + i];
                for (int k = 0; k < n; k++) {
                    A[j * n + k] -= factor * A[i * n + k];
                    inverse[j * n + k] -= factor * inverse[i * n + k];
                }
            }
        }
    }
}

void multiplyMatrices(REAL *A, REAL *B, REAL *result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                result[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void subtractMatrices(REAL *A, REAL *B, REAL *result, int n) {
    for (int i = 0; i < n * n; i++) {
        result[i] = A[i] - B[i];
    }
}

void copyMatrix(REAL *src, REAL *dest, int n) {
    for (int i = 0; i < n * n; i++) {
        dest[i] = src[i];
    }
}

