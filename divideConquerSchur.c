#include <stdio.h>
#include <stdlib.h>

void printMatrix(double** matrix, int size);
double** allocateMatrix(int size);
void freeMatrix(double** matrix, int size);
void multiply(double** A, double** B, double** result, int size);
void subtract(double** A, double** B, double** result, int size);
void invert2x2(double** A, double** result);
void copySubmatrix(double** src, double** dest, int srcRow, int srcCol, int size);
void setSubmatrix(double** dest, double** src, int destRow, int destCol, int size);
void divideAndConquerInvert(double** F, double** result, int size);

// Main function to demonstrate matrix inversion
int main() {
    int size = 4;
    double** F = allocateMatrix(size);
    double** F_inv = allocateMatrix(size);

    srand(0u);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
	    double n = rand();
            F[i][j] = ((double)n==0?1:n/RAND_MAX);
        }
    divideAndConquerInvert(F, F_inv, size);

    double sum = 0.0f;
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
          sum += F_inv[i][j];
    freeMatrix(F, size);
    freeMatrix(F_inv, size);
    printf("Checksum: %.8f\n", sum);
    return 0;
}

// Function to print a square matrix
void printMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Allocate memory for a matrix of given size
double** allocateMatrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    return matrix;
}

// Free allocated memory of a matrix
void freeMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Direct inversion for a 2x2 matrix
void invert2x2(double** A, double** result) {
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    result[0][0] = A[1][1] / det;
    result[0][1] = -A[0][1] / det;
    result[1][0] = -A[1][0] / det;
    result[1][1] = A[0][0] / det;
}

// Recursive divide-and-conquer matrix inversion
void divideAndConquerInvert(double** F, double** result, int size) {
    if (size == 1) {
        result[0][0] = 1.0 / F[0][0];
        return;
    } else if (size == 2) {
        invert2x2(F, result);
        return;
    }

    int halfSize = size / 2;
    double** A = allocateMatrix(halfSize);
    double** B = allocateMatrix(halfSize);
    double** C = allocateMatrix(halfSize);
    double** A_inv = allocateMatrix(halfSize);
    double** S = allocateMatrix(halfSize);
    double** S_inv = allocateMatrix(halfSize);

    // Step 1: Partition F into submatrices A, B, C
    copySubmatrix(F, A, 0, 0, halfSize);
    copySubmatrix(F, B, 0, halfSize, halfSize);
    copySubmatrix(F, C, halfSize, halfSize, halfSize);

    // Step 2: Invert A recursively
    divideAndConquerInvert(A, A_inv, halfSize);

    // Step 3: Compute Schur complement S = C - B^T * A_inv * B
    double** BA_inv = allocateMatrix(halfSize);
    double** BA_invB = allocateMatrix(halfSize);
    multiply(B, A_inv, BA_inv, halfSize);       // BA_inv = B * A_inv
    multiply(BA_inv, B, BA_invB, halfSize);     // BA_invB = BA_inv * B
    subtract(C, BA_invB, S, halfSize);          // S = C - BA_invB

    // Step 4: Invert S recursively
    divideAndConquerInvert(S, S_inv, halfSize);

    // Step 5: Compute result blocks
    double** temp1 = allocateMatrix(halfSize);
    double** temp2 = allocateMatrix(halfSize);

    multiply(BA_inv, S_inv, temp1, halfSize);         // temp1 = BA_inv * S_inv
    multiply(S_inv, B, temp2, halfSize);              // temp2 = S_inv * B

    for (int i = 0; i < halfSize; i++) {
        for (int j = 0; j < halfSize; j++) {
            result[i][j] = A_inv[i][j] + temp1[i][j] * B[i][j];
            result[i][j + halfSize] = -temp1[i][j];
            result[i + halfSize][j] = -temp2[i][j];
            result[i + halfSize][j + halfSize] = S_inv[i][j];
        }
    }

    // Free allocated submatrices
    freeMatrix(A, halfSize);
    freeMatrix(B, halfSize);
    freeMatrix(C, halfSize);
    freeMatrix(A_inv, halfSize);
    freeMatrix(S, halfSize);
    freeMatrix(S_inv, halfSize);
    freeMatrix(BA_inv, halfSize);
    freeMatrix(BA_invB, halfSize);
    freeMatrix(temp1, halfSize);
    freeMatrix(temp2, halfSize);
}

// Helper functions to work with submatrices
void copySubmatrix(double** src, double** dest, int srcRow, int srcCol, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            dest[i][j] = src[srcRow + i][srcCol + j];
}

void setSubmatrix(double** dest, double** src, int destRow, int destCol, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            dest[destRow + i][destCol + j] = src[i][j];
}

// Matrix multiplication
void multiply(double** A, double** B, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix subtraction
void subtract(double** A, double** B, double** result, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            result[i][j] = A[i][j] - B[i][j];
}
