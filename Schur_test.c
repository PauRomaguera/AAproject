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

void GJElimination(double *input, double *inverse, int N) {
//  double *inverse = (double *)malloc(N * N * sizeof(double));
  double pivot, pivot2, sum;
  int i, j, k;


  // initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (i == j)
        inverse[i*N+j] = 1.0f;
      else
        inverse[i*N+j] = 0.0f;


  // computes the inverse matrix
  for (i = 0; i < N; i++)
  {
    pivot = input[i*N+i];
    for (j = 0; j < N; j++)
    {
      input[i*N+j] /= pivot;
      inverse[i*N+j] /= pivot;
    }
    for (j = 0; j < N; j++)
      if (i != j)
      {
        pivot2 = input[j*N+i];
        for (k = 0; k < N; k++)
          input[j * N + k] -= input[i * N + k] * pivot2;
        for (k = 0; k < N; k++)
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
      }
  }
  //return inverse;
}

// Recursive function to invert a matrix using the Schur complement
void schur_inverse(double *A, double *A_inv, int n, int base_case_size) {
    if (n <= base_case_size) {
        if (n == 1) {
            A_inv[0] = 1 / A[0];
            return;
        }
        GJElimination(A, A_inv, n);
        return;
    }

    int half = n / 2;

    // Define submatrices as views
    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + half * n;
    double *A22 = A + half * n + half;

    double *A11_inv = A_inv;
    double *A12_inv = A_inv + half;
    double *A21_inv = A_inv + half * n;
    double *A22_inv = A_inv + half * n + half;

    // Recursive call for A11_inv
    schur_inverse(A11, A11_inv, half, base_case_size);

    // Allocate memory for temporary matrices only when needed
    double *temp1 = (double *)malloc(half * half * sizeof(double));
    double *temp2 = (double *)malloc(half * half * sizeof(double));
    double *S = (double *)malloc(half * half * sizeof(double));
    double *S_inv = (double *)malloc(half * half * sizeof(double));

    // Compute Schur complement S = A22 - A21 * A11_inv * A12
    matmul(A21, A11_inv, temp1, half, half, half); // temp1 = A21 * A11_inv
    matmul(temp1, A12, temp2, half, half, half);   // temp2 = temp1 * A12
    mat_sub(A22, temp2, S, half);                 // S = A22 - temp2

    // Recursive call for S_inv
    schur_inverse(S, S_inv, half, base_case_size);

    // Compute blocks of A_inv using index-based access
    matmul(A11_inv, A12, temp1, half, half, half); // temp1 = A11_inv * A12
    matmul(temp1, S_inv, temp2, half, half, half); // temp2 = temp1 * S_inv
    matmul(temp2, A21, temp1, half, half, half);   // temp1 = temp2 * A21
    matmul(temp1, A11_inv, temp2, half, half, half); // temp2 = temp1 * A11_inv
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11_inv[i * half + j] += temp2[i * half + j]; // A_inv11 = A11_inv + temp2
        }
    }

    matmul(A11_inv, A12, temp1, half, half, half);  // temp1 = A11_inv * A12
    matmul(temp1, S_inv, A12_inv, half, half, half); // A_inv12 = -temp1
    for (int i = 0; i < half * half; i++) {
        A12_inv[i] = -A12_inv[i];
    }

    matmul(S_inv, A21, temp1, half, half, half);   // temp1 = S_inv * A21
    matmul(temp1, A11_inv, A21_inv, half, half, half); // A_inv21 = -temp1
    for (int i = 0; i < half * half; i++) {
        A21_inv[i] = -A21_inv[i];
    }

    // Copy S_inv to A_inv22 directly
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A22_inv[i * half + j] = S_inv[i * half + j];
        }
    }

    // Free allocated memory
    free(temp1);
    free(temp2);
    free(S);
    free(S_inv);
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <base_case_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int base_case_size = atoi(argv[2]);

    if (n <= 0 || base_case_size <= 0 || n % base_case_size != 0) {
        printf("Error: Ensure matrix_size > 0, base_case_size > 0, and matrix_size is divisible by base_case_size.\n");
        return 1;
    }
    double *A = (double *)malloc(n * n * sizeof(double));
    double *A_inv = (double *)malloc(n * n * sizeof(double));
    srand(0u);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A[i*n+j] = ((double)rand()/RAND_MAX);

    schur_inverse(A, A_inv, n, base_case_size);
//    print_matrix(A_inv, n);
    double sum = 0.0f;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        sum += A_inv[i * n + j];

    printf("Checksum A:%.8f \n", sum);

    return 0;
}
