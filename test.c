#include <stdio.h>
#include <stdlib.h>

#define N 1500  // Matrix size
#define REAL double
#define DQSZ 2  // Base case size (2x2)

// Base case for Gauss-Jordan elimination on small matrices
void GJ_Base(REAL* input, REAL* inverse, int SZ) {
    int i, j, k;
    REAL pivot, pivot2;
    for (i = 0; i < SZ; i++) {
        pivot = input[i * N + i];
        for (j = 0; j < SZ; j++) {
            input[i * N + j] /= pivot;
            inverse[i * N + j] /= pivot;
        }
        for (j = 0; j < SZ; j++) {
            if (i != j) {
                pivot2 = input[j * N + i];
                for (k = 0; k < SZ; k++) {
                    input[j * N + k] -= input[i * N + k] * pivot2;
                    inverse[j * N + k] -= inverse[i * N + k] * pivot2;
                }
            }
        }
    }
}

// Matrix subtraction: result = A - B
void matrix_subtract(REAL* A, REAL* B, REAL* result, int SZ) {
    for (int i = 0; i < SZ; i++) {
        for (int j = 0; j < SZ; j++) {
            result[i * N + j] = A[i * N + j] - B[i * N + j];
        }
    }
}

// Matrix multiplication: result = A * B
void matrix_multiply(REAL* A, REAL* B, REAL* result, int SZ) {
    for (int i = 0; i < SZ; i++) {
        for (int j = 0; j < SZ; j++) {
            result[i * N + j] = 0;
            for (int k = 0; k < SZ; k++) {
                result[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// Divide and conquer for Gauss-Jordan matrix inversion
void GaussJordan_DQ(REAL* input, REAL* inverse, int SZ) {
    if (SZ <= DQSZ) {
        // Base case: use Gauss-Jordan elimination for small matrices
        GJ_Base(input, inverse, SZ);
        return;
    }

    // Reduce the size by half
    SZ = SZ / 2;

    // Block pointers
    REAL *A11 = input, *A12 = input + SZ, *A21 = input + SZ * N, *A22 = input + SZ * (N + 1);
    REAL *I11 = inverse, *I12 = inverse + SZ, *I21 = inverse + SZ * N, *I22 = inverse + SZ * (N + 1);

    // Temporary storage for intermediate results
    REAL* temp = (REAL*)malloc(N * N * sizeof(REAL));

    // Step 1: Invert A11
    GaussJordan_DQ(A11, I11, SZ);

    // Step 2: Compute A12 = A11^-1 * A12
    matrix_multiply(I11, A12, temp, SZ);
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            A12[i * N + j] = temp[i * N + j];

    // Step 3: Compute A21 = A21 * A11^-1
    matrix_multiply(A21, I11, temp, SZ);
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            A21[i * N + j] = temp[i * N + j];

    // Step 4: Compute Schur complement: A22 = A22 - A21 * A12
    matrix_multiply(A21, A12, temp, SZ);
    matrix_subtract(A22, temp, A22, SZ);

    // Step 5: Invert Schur complement (A22)
    GaussJordan_DQ(A22, I22, SZ);

    // Free temporary storage
    free(temp);
}

int main() {
    printf("Matrix size: %dx%d\n", N, N);
    REAL sum = 0;
    REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
    REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));

    // Initialize matrix with pseudo-random numbers (always the same)
    srand(0u);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            input[i * N + j] = ((REAL)rand() / RAND_MAX);

    // Initialize identity matrix
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i == j)
                inverse[i * N + j] = 1.0;
            else
                inverse[i * N + j] = 0.0;

    // Perform divide-and-conquer Gauss-Jordan inversion
    GaussJordan_DQ(input, inverse, N);

    // Compute checksum
    sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += inverse[i * N + j];

    printf("Checksum: %.8f\n", sum);

    free(input);
    free(inverse);

    return 0;
}
