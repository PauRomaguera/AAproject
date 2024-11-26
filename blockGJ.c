#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 1500  // Size of the matrix (1500x1500)
#define BLOCK_SIZE 500  // Define submatrix block size

void gauss_jordan_block(double A[SIZE*SIZE], double B[SIZE*SIZE], int startRow, int startCol, int blockSize) {
    // Perform Gauss-Jordan elimination on the block starting from (startRow, startCol) with size blockSize
    for (int k = startRow; k < startRow + blockSize; k++) {
        // Normalize pivot row
        double pivot = A[k*SIZE+k];
        for (int j = startCol; j < startCol + blockSize; j++) {
            A[k*SIZE+j] /= pivot;
            B[k*SIZE+j] /= pivot;
        }

        // Eliminate other rows
	int i, j;
	double factor;
        #pragma omp parallel for private(j, i, factor) shared(A, B, k)
        for (i = startRow; i < startRow + blockSize; i++) {
            if (i != k) {
                factor = A[i*SIZE+k];
                for (j = startCol; j < startCol + blockSize; j++) {
                    A[i*SIZE+j] -= factor * A[k*SIZE+j];
                    B[i*SIZE+j] -= factor * B[k*SIZE+j];
                }
            }
        }
    }
}


void update_row_block(double A[SIZE*SIZE], double B[SIZE*SIZE], int rowBlock, int pivotBlock, int blockSize) {
    // Update the row blocks (A[rowBlock][*]) using the pivot block (A[pivotBlock][*])
    for (int i = rowBlock; i < rowBlock + blockSize; i++) {
        for (int j = pivotBlock; j < pivotBlock + blockSize; j++) {
            if (i >= pivotBlock + blockSize) {
                double factor = A[i*SIZE+pivotBlock];
                for (int k = pivotBlock; k < pivotBlock + blockSize; k++) {
                    A[i*SIZE+k] -= factor * A[pivotBlock*SIZE+k];
                    B[i*SIZE+k] -= factor * B[pivotBlock*SIZE+k];
                }
            }
        }
    }
}

void update_column_block(double A[SIZE*SIZE], double B[SIZE*SIZE], int colBlock, int pivotBlock, int blockSize) {
    // Update the column blocks (A[*][colBlock]) using the pivot block (A[*][pivotBlock])
    for (int i = pivotBlock; i < pivotBlock + blockSize; i++) {
        for (int j = colBlock; j < colBlock + blockSize; j++) {
            if (j >= pivotBlock + blockSize) {
                double factor = A[pivotBlock*SIZE+j];
                for (int k = colBlock; k < colBlock + blockSize; k++) {
                    A[i*SIZE+k] -= factor * A[i*SIZE+pivotBlock];
                    B[i*SIZE+k] -= factor * B[i*SIZE+pivotBlock];
                }
            }
        }
    }
}

void update_off_diagonal_blocks(double A[SIZE*SIZE], double B[SIZE*SIZE], int rowBlock, int colBlock, int pivotBlock, int blockSize) {
    for (int i = rowBlock; i < rowBlock + blockSize; i++) {
        for (int j = colBlock; j < colBlock + blockSize; j++) {
//            if (i!=j) {
            for (int i = rowBlock; i < rowBlock + blockSize; i++) {
                for (int j = colBlock; j < colBlock + blockSize; j++) {
            double factor = A[i*SIZE+pivotBlock];
            for (int k = pivotBlock; k < pivotBlock + blockSize; k++) {
                A[i*SIZE+j] -= factor * A[k*SIZE+j];
                B[i*SIZE+j] -= factor * B[k*SIZE+j];
            }
//	    }
}}
        }
    }
}

void gauss_jordan(double A[SIZE*SIZE], double B[SIZE*SIZE]) {
    // Initialize B as an identity matrix
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            B[i*SIZE+j] = (i == j) ? 1 : 0;

    // Divide the matrix into blocks and apply Gauss-Jordan to each block
    for (int pivotBlock = 0; pivotBlock < SIZE; pivotBlock += BLOCK_SIZE) {
        // 1. Perform Gauss-Jordan elimination on the current pivot block
        gauss_jordan_block(A, B, pivotBlock, pivotBlock, BLOCK_SIZE);

        // 2. Update the row blocks below the pivot block
        for (int rowBlock = pivotBlock + BLOCK_SIZE; rowBlock < SIZE; rowBlock += BLOCK_SIZE) {
            update_row_block(A, B, rowBlock, pivotBlock, BLOCK_SIZE);
        }

        // 3. Update the column blocks to the right of the pivot block
        for (int colBlock = pivotBlock + BLOCK_SIZE; colBlock < SIZE; colBlock += BLOCK_SIZE) {
            update_column_block(A, B, colBlock, pivotBlock, BLOCK_SIZE);
        }

        for (int rowBlock = pivotBlock + BLOCK_SIZE; rowBlock < SIZE; rowBlock += BLOCK_SIZE) {
            for (int colBlock = pivotBlock + BLOCK_SIZE; colBlock < SIZE; colBlock += BLOCK_SIZE) {
                update_off_diagonal_blocks(A, B, rowBlock, colBlock, pivotBlock, BLOCK_SIZE);
            }
        }
    }

    // Further steps for inter-block combinations can be added here for larger scenarios
}

int main() {
//    double (*A)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));
//    double (*B)[SIZE] = malloc(sizeof(double[SIZE][SIZE]));
    double *A = (double *)malloc(SIZE * SIZE * sizeof(double));
    double *B = (double *)malloc(SIZE * SIZE * sizeof(double));

    srand(0u);
    // Initialize matrix A (example initialization)
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            A[i*SIZE+j] = ((double)rand()/RAND_MAX);

    gauss_jordan(A, B);

    double sum = 0.0f;
    for (int i = 0; i < SIZE; i++)
      for (int j = 0; j < SIZE; j++)
        sum += B[i * SIZE + j];
    free(A);
    free(B);
    printf("Checksum: %.8f\n", sum);
    return 0;
}

