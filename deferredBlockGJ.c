#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define SIZE 1500  // Size of the matrix (1500x1500)
#define BLOCK_SIZE 500  // Define submatrix block size

// Function to perform Gauss-Jordan elimination on a pivot block with deferred updates
void process_pivot_block(double tempA[SIZE][SIZE], double tempB[SIZE][SIZE], int pivot, int blockSize) {
    for (int k = pivot; k < pivot + blockSize; k++) {
        double pivot_value = tempA[k][k];

        // Normalize the pivot row
        for (int j = 0; j < SIZE; j++) {
            tempA[k][j] /= pivot_value;
            tempB[k][j] /= pivot_value;
        }

        // Eliminate other rows in the pivot block
        for (int i = 0; i < SIZE; i++) {
            if (i != k) {
                double factor = tempA[i][k];
                for (int j = 0; j < SIZE; j++) {
                    tempA[i][j] -= factor * tempA[k][j];
                    tempB[i][j] -= factor * tempB[k][j];
                }
            }
        }
    }
}

// Function to process a row block with deferred propagation
void process_row_block(double tempA[SIZE][SIZE], double tempB[SIZE][SIZE], int rowBlock, int pivot, int blockSize) {
    for (int i = rowBlock; i < rowBlock + blockSize; i++) {
        if (i >= pivot + blockSize) {  // Ensure we skip rows in the pivot block
            double factor = tempA[i][pivot];
            for (int j = pivot; j < SIZE; j++) {
                tempA[i][j] -= factor * tempA[pivot][j];
                tempB[i][j] -= factor * tempB[pivot][j];
            }
        }
    }
}

// Function to process a column block with deferred propagation
void process_column_block(double tempA[SIZE][SIZE], double tempB[SIZE][SIZE], int colBlock, int pivot, int blockSize) {
    for (int j = colBlock; j < colBlock + blockSize; j++) {
        if (j >= pivot + blockSize) {  // Ensure we skip columns in the pivot block
            double factor = tempA[pivot][j];
            for (int i = pivot + 1; i < SIZE; i++) {
                tempA[i][j] -= factor * tempA[i][pivot];
                tempB[i][j] -= factor * tempB[i][pivot];
            }
        }
    }
}

// Function to process an off-diagonal block with deferred propagation
void process_off_diagonal_block(double tempA[SIZE][SIZE], double tempB[SIZE][SIZE], int rowBlock, int colBlock, int pivot, int blockSize) {
    for (int i = rowBlock; i < rowBlock + blockSize; i++) {
        for (int j = colBlock; j < colBlock + blockSize; j++) {
            double factor = tempA[i][pivot];
            for (int k = pivot; k < pivot + blockSize; k++) {
                tempA[i][j] -= factor * tempA[k][j];
                tempB[i][j] -= factor * tempB[k][j];
            }
        }
    }
}

// Function to propagate all accumulated updates at the end of the iteration
void propagate_changes(double A[SIZE][SIZE], double B[SIZE][SIZE], double tempA[SIZE][SIZE], double tempB[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = tempA[i][j];
            B[i][j] = tempB[i][j];
        }
    }
}

void gauss_jordan(double A[SIZE][SIZE], double B[SIZE][SIZE]) {
    // Initialize B as an identity matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            B[i][j] = (i == j) ? 1 : 0;
        }
    }

    // Temporary matrices for deferred propagation
    double (*tempA)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));
    double (*tempB)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));

    // Copy original matrices to tempA and tempB for deferred updates
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            tempA[i][j] = A[i][j];
            tempB[i][j] = B[i][j];
        }
    }

    // Process blocks
    for (int pivotBlock = 0; pivotBlock < SIZE; pivotBlock += BLOCK_SIZE) {
        // Step 1: Process the pivot block
        process_pivot_block(tempA, tempB, pivotBlock, BLOCK_SIZE);

        // Step 2: Process row and column blocks without immediate propagation
        for (int rowBlock = pivotBlock + BLOCK_SIZE; rowBlock < SIZE; rowBlock += BLOCK_SIZE) {
            process_row_block(tempA, tempB, rowBlock, pivotBlock, BLOCK_SIZE);
        }
        for (int colBlock = pivotBlock + BLOCK_SIZE; colBlock < SIZE; colBlock += BLOCK_SIZE) {
            process_column_block(tempA, tempB, colBlock, pivotBlock, BLOCK_SIZE);
        }

        // Step 3: Process off-diagonal blocks
        for (int rowBlock = pivotBlock + BLOCK_SIZE; rowBlock < SIZE; rowBlock += BLOCK_SIZE) {
            for (int colBlock = pivotBlock + BLOCK_SIZE; colBlock < SIZE; colBlock += BLOCK_SIZE) {
                process_off_diagonal_block(tempA, tempB, rowBlock, colBlock, pivotBlock, BLOCK_SIZE);
            }
        }

        // Step 4: Propagate all changes to tempA and tempB (apply deferred updates)
        propagate_changes(A, B, tempA, tempB);
    }

    free(tempA);
    free(tempB);
}

int main() {
    double (*A)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));
    double (*B)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));

    srand(0u);
    // Initialize matrix A (example initialization)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = ((double)rand() / RAND_MAX);
        }
    }

    gauss_jordan(A, B);

  // Computes checksum
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; i++)
      for (int j = 0; j < SIZE; j++)
        sum += B[i][j];

    free(A);
    free(B);
    printf("Checksum: %.8f\n", sum);
    return 0;
}

