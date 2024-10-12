#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define REAL float
#define N 1500

// Perform LU decomposition with partial pivoting
void lu_decomposition(REAL *A, int *P) {
    int i, j, k, max_index;
    REAL max_value, temp;
    
    // Initialize permutation vector P as identity
    for (i = 0; i < N; i++) {
        P[i] = i;
    }

    for (k = 0; k < N; k++) {
        // Find the row with the largest pivot element
        max_index = k;
        max_value = fabs(A[k * N + k]);
        for (i = k + 1; i < N; i++) {
            if (fabs(A[i * N + k]) > max_value) {
                max_value = fabs(A[i * N + k]);
                max_index = i;
            }
        }

        // Swap rows if necessary
        if (max_index != k) {
            for (j = 0; j < N; j++) {
                temp = A[k * N + j];
                A[k * N + j] = A[max_index * N + j];
                A[max_index * N + j] = temp;
            }

            // Swap the permutation vector
            temp = P[k];
            P[k] = P[max_index];
            P[max_index] = temp;
        }

        // Perform Gaussian elimination
        for (i = k + 1; i < N; i++) {
            REAL factor = A[i * N + k] / A[k * N + k];
            A[i * N + k] = factor;
            for (j = k + 1; j < N; j++) {
                A[i * N + j] -= factor * A[k * N + j];
            }
        }
    }
}

// Forward substitution to solve L*y = b
void forward_substitution(REAL *A, REAL *b, REAL *y, int *P) {
    int i, j;

    for (i = 0; i < N; i++) {
        y[i] = b[P[i]];
        for (j = 0; j < i; j++) {
            y[i] -= A[i * N + j] * y[j];
        }
    }
}

// Backward substitution to solve U*x = y
void backward_substitution(REAL *A, REAL *y, REAL *x) {
    int i, j;

    for (i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (j = i + 1; j < N; j++) {
            x[i] -= A[i * N + j] * x[j];
        }
        x[i] /= A[i * N + i];
    }
}

int main() {
    printf("Matrix size: %dx%d\n", N, N);

    REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
    REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
    REAL *b = (REAL *)malloc(N * sizeof(REAL));
    REAL *y = (REAL *)malloc(N * sizeof(REAL));
    int *P = (int *)malloc(N * sizeof(int));
    REAL sum;
    int i, j, k;

    // Initialize the matrix with pseudo-random numbers
    srand(0u);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            input[i * N + j] = ((REAL)rand() / RAND_MAX);
        }
    }

    // Initialize the identity matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j)
                inverse[i * N + j] = 1.0;
            else
                inverse[i * N + j] = 0.0;
        }
    }

    // Perform LU decomposition with partial pivoting
    lu_decomposition(input, P);

    // Solve for the inverse using forward and backward substitution
    for (k = 0; k < N; k++) {
        // Initialize b as the k-th column of the identity matrix
        for (i = 0; i < N; i++) {
            b[i] = (i == k) ? 1.0 : 0.0;
        }

        // Forward substitution (L * y = b)
        forward_substitution(input, b, y, P);

        // Backward substitution (U * x = y)
        backward_substitution(input, y, &inverse[k * N]);
    }

    // Compute checksum for the inverse matrix
    sum = 0.0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum += inverse[i * N + j];
        }
    }

    printf("Checksum: %.8f\n", sum);

    // Free allocated memory
    free(input);
    free(inverse);
    free(b);
    free(y);
    free(P);

    return 0;
}
