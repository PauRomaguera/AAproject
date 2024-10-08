#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define REAL double
#define N 1500

// Function to perform Cholesky decomposition
int cholesky_decomposition(REAL* matrix, REAL* L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            REAL sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i * n + k] * L[j * n + k];
            if (i == j)
                L[i * n + i] = sqrt(matrix[i * n + i] - sum);
            else
                L[i * n + j] = (matrix[i * n + j] - sum) / L[j * n + j];
        }
    }
    return 1;
}

// Function to invert lower triangular matrix L
void invert_lower_triangular(REAL* L, REAL* Linv, int n) {
    for (int i = 0; i < n; i++) {
        Linv[i * n + i] = 1 / L[i * n + i]; // Diagonal elements
        for (int j = 0; j < i; j++) {
            REAL sum = 0;
            for (int k = j; k < i; k++)
                sum += L[i * n + k] * Linv[k * n + j];
            Linv[i * n + j] = -sum / L[i * n + i];
        }
    }
}

// Function to invert matrix using Cholesky decomposition
void cholesky_inversion(REAL* matrix, REAL* inverse, int n) {
    REAL* L = (REAL*)malloc(n * n * sizeof(REAL));
    REAL* Linv = (REAL*)malloc(n * n * sizeof(REAL));

    // Perform Cholesky decomposition
    if (!cholesky_decomposition(matrix, L, n)) {
        free(L);
        free(Linv);
        return;
    }

    // Invert the lower triangular matrix L
    invert_lower_triangular(L, Linv, n);

    // Compute the inverse of the original matrix as Linv^T * Linv
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            REAL sum = 0;
            for (int k = i > j ? i : j; k < n; k++) {
                sum += Linv[k * n + i] * Linv[k * n + j];
            }
            inverse[i * n + j] = sum;
        }
    }

    free(L);
    free(Linv);
}

int main() {
    printf("Matrix size: %dx%d\n", N, N);

    REAL* input = (REAL*)malloc(N * N * sizeof(REAL));
    REAL* inverse = (REAL*)malloc(N * N * sizeof(REAL));
    REAL sum;
    int i, j;

    // Initializes matrix with pseudo-random symmetric positive definite numbers
    srand(0u);
    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            input[i * N + j] = ((REAL)rand() / RAND_MAX);
            input[j * N + i] = input[i * N + j]; // Make the matrix symmetric
        }
        input[i * N + i] += N; // Ensure positive definiteness
    }

    // Perform matrix inversion using Cholesky decomposition
    cholesky_inversion(input, inverse, N);

    // Compute checksum for verification
    sum = 0.0;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            sum += inverse[i * N + j];

    printf("Checksum: %.8f\n", sum);

    free(input);
    free(inverse);
    return 0;
}
