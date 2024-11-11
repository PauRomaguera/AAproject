#include <stdio.h>
#include <stdlib.h>

// Helper function to perform matrix multiplication
void matmul(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Helper function to subtract two matrices
void mat_sub(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Function to print a matrix
void print_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Recursive function to invert a matrix using the Schur complement
void schur_inverse(double *A, double *A_inv, int n) {
    if (n == 1) {
        // Base case: inverse of 1x1 matrix
        A_inv[0] = 1 / A[0];
        return;
    }

    int half = n / 2;

    // Allocate memory for blocks
    double *A11 = (double *)malloc(half * half * sizeof(double));
    double *A12 = (double *)malloc(half * half * sizeof(double));
    double *A21 = (double *)malloc(half * half * sizeof(double));
    double *A22 = (double *)malloc(half * half * sizeof(double));

    double *A11_inv = (double *)malloc(half * half * sizeof(double));
    double *S = (double *)malloc(half * half * sizeof(double));
    double *S_inv = (double *)malloc(half * half * sizeof(double));

    // Split A into blocks
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i * half + j] = A[i * n + j];
            A12[i * half + j] = A[i * n + j + half];
            A21[i * half + j] = A[(i + half) * n + j];
            A22[i * half + j] = A[(i + half) * n + j + half];
        }
    }
/*
    printf("A11:\n");
    print_matrix(A11, half);
    printf("A12:\n");
    print_matrix(A12, half);
    printf("A21:\n");
    print_matrix(A21, half);
    printf("A22:\n");
    print_matrix(A22, half);
*/
    // Recursive call for A11_inv
    schur_inverse(A11, A11_inv, half);
  //  printf("A11_inv:\n");
  //  print_matrix(A11_inv, half);

    // Compute Schur complement S = A22 - A21 * A11_inv * A12
    double *temp1 = (double *)malloc(half * half * sizeof(double));
    double *temp2 = (double *)malloc(half * half * sizeof(double));
    matmul(A21, A11_inv, temp1, half, half, half);
    matmul(temp1, A12, temp2, half, half, half);
    mat_sub(A22, temp2, S, half);

  //  printf("Schur Complement S:\n");
  //  print_matrix(S, half);

    // Recursive call for S_inv
    schur_inverse(S, S_inv, half);
  //  printf("S_inv:\n");
  //  print_matrix(S_inv, half);

    // Compute the inverse of the original matrix A using block inversion formula
    double *A_inv11 = (double *)malloc(half * half * sizeof(double));
    double *A_inv12 = (double *)malloc(half * half * sizeof(double));
    double *A_inv21 = (double *)malloc(half * half * sizeof(double));
    double *A_inv22 = (double *)malloc(half * half * sizeof(double));

    // A_inv11 = A11_inv + A11_inv * A12 * S_inv * A21 * A11_inv
    matmul(A11_inv, A12, temp1, half, half, half);
    matmul(temp1, S_inv, temp2, half, half, half);
    matmul(temp2, A21, temp1, half, half, half);
    matmul(temp1, A11_inv, A_inv11, half, half, half);

//    matmul(A12, S_inv, temp1, half, half, half);
//    matmul(temp1, A21, temp2, half, half, half);
//    matmul(A11_inv, temp2, A_inv11, half, half, half);

    for (int i = 0; i < half * half; i++) {
        A_inv11[i] += A11_inv[i];
    }

  //  printf("A_inv11:\n");
  //  print_matrix(A_inv11, half);

    // A_inv12 = -A11_inv * A12 * S_inv
    matmul(A11_inv, A12, temp1, half, half, half);
    matmul(temp1, S_inv, A_inv12, half, half, half);
    for (int i = 0; i < half * half; i++) {
        A_inv12[i] = -A_inv12[i];
    }

  //  printf("A_inv12:\n");
  //  print_matrix(A_inv12, half);

    // A_inv21 = -S_inv * A21 * A11_inv
    matmul(S_inv, A21, temp1, half, half, half);
    matmul(temp1, A11_inv, A_inv21, half, half, half);
    for (int i = 0; i < half * half; i++) {
        A_inv21[i] = -A_inv21[i];
    }

  //  printf("A_inv21:\n");
  //  print_matrix(A_inv21, half);

    // A_inv22 = S_inv
    for (int i = 0; i < half * half; i++) {
        A_inv22[i] = S_inv[i];
    }

  //  printf("A_inv22:\n");
  //  print_matrix(A_inv22, half);

    // Combine blocks into A_inv
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A_inv[i * n + j] = A_inv11[i * half + j];
            A_inv[i * n + j + half] = A_inv12[i * half + j];
            A_inv[(i + half) * n + j] = A_inv21[i * half + j];
            A_inv[(i + half) * n + j + half] = A_inv22[i * half + j];
        }
    }

    // Free allocated memory
    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(A11_inv);
    free(S);
    free(S_inv);
    free(temp1);
    free(temp2);
    free(A_inv11);
    free(A_inv12);
    free(A_inv21);
    free(A_inv22);
}

int main() {
    int n = 2048;
    double *A = (double *)malloc(n * n * sizeof(double));
    double *A_inv = (double *)malloc(n * n * sizeof(double));
    srand(0u);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A[i*n+j] = ((double)rand()/RAND_MAX);

    schur_inverse(A, A_inv, n);
//    print_matrix(A_inv, n);
    double sum = 0.0f;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        sum += A_inv[i * n + j];

    printf("Checksum A:%1f", sum);

    return 0;
}
