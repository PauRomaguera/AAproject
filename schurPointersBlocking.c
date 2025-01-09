#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int N = 4;
#include <stddef.h> // per a size_t
#define BS 16  // Bloc de 64x64

void matmul_blocked(const double *A, size_t ldA,
                    const double *B, size_t ldB,
                    double *C,    size_t ldC,
                    size_t M,     size_t N, size_t K)
{
    // Tria una mida de bloc (64 és un exemple, pots provar 32, 128, etc.)    
    for (size_t i = 0; i < M; i += BS) {
        for (size_t j = 0; j < N; j += BS) {
            for (size_t kb = 0; kb < K; kb += BS) {

                // Treballem amb subblocs A[i..i+BS-1, kb..kb+BS-1]
                //                     B[kb..kb+BS-1, j..j+BS-1]
                // i actualitzem C[i..i+BS-1, j..j+BS-1]
                for (size_t ii = i; ii < i + BS && ii < M; ii++) {
                    for (size_t jj = j; jj < j + BS && jj < N; jj++) {

                        double sum = 0.0;
                        for (size_t kk = kb; kk < kb + BS && kk < K; kk++) {
                            sum += A[ii * ldA + kk] * B[kk * ldB + jj];
                        }
                        // Acumulem resultats a C
                        C[ii * ldC + jj] += sum;
                    }
                }
            }
        }
    }
}


void matmul(const double *A, size_t ldA,
            const double *B, size_t ldB,
            double *C, size_t ldC,
            size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * ldA + k] * B[k * ldB + j];
            }
            C[i * ldC + j] = sum;
        }
    }
}

// Agafa mat A tamany N, guarda en mat A_inv tamany size
void inverse_2x2_stride(double *A, int ldA, double *A_inv, int ldInv)
{
    // printf("[DEBUG] Entering inverse_2x2_stride()\n");
    // printf("[DEBUG] ldA=%d, ldInv=%d\n", ldA, ldInv);

    // Indices basados en ldA
    double a = A[0 * ldA + 0];
    double b = A[0 * ldA + 1];
    double c = A[1 * ldA + 0];
    double d = A[1 * ldA + 1];

    // printf("[DEBUG] Sub-bloque 2x2:\n");
    // printf("   A[0,0]=%.6f, A[0,1]=%.6f\n", a, b);
    // printf("   A[1,0]=%.6f, A[1,1]=%.6f\n", c, d);

    double det = a * d - b * c;
    // printf("[DEBUG] Determinante: %.6f\n", det);
    if (det == 0.0)
    {
        fprintf(stderr, "[ERROR] 2x2 block is singular.\n");
        exit(EXIT_FAILURE);
    }

    // Escribimos la inversa en A_inv, usando ldInv
    double inv_a = d / det;
    double inv_b = -b / det;
    double inv_c = -c / det;
    double inv_d = a / det;

    A_inv[0 * ldInv + 0] = inv_a;
    A_inv[0 * ldInv + 1] = inv_b;
    A_inv[1 * ldInv + 0] = inv_c;
    A_inv[1 * ldInv + 1] = inv_d;

    // printf("[DEBUG] Inversa 2x2:\n");
    // printf("   A_inv[0,0]=%.6f, A_inv[0,1]=%.6f\n", inv_a, inv_b);
    // printf("   A_inv[1,0]=%.6f, A_inv[1,1]=%.6f\n\n", inv_c, inv_d);
}
/**
 * GJElimination_stride:
 *  Fa Gauss-Jordan per a un bloc de mida n×n contingut a la matriu 'A'
 *  (sub-bloc amb stride ldA).
 *  Escriu el resultat en 'A_inv', que és un buffer contigu (n×n).
 *
 * Paràmetres:
 *   A     : Punter al bloc (n×n) incrustat, amb stride = ldA
 *   ldA   : Leading dimension (distància entre files) de A
 *   A_inv : Buffer de mida n×n (contigu) on es guardarà la inversa
 *   n     : Dimensió lògica del bloc (n×n)
 */
void GJElimination_stride(double *A, int ldA, double *A_inv, int n)
{
    // printf("--------------\n");
    // printf("GAUSS JORDAN (stride)\n");
    // printf("--------------\n");

    // Inicialitza la sortida A_inv com a matriu identitat (mida n×n contigu)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A_inv[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (int i = 0; i < n; i++)
    {
        // 1) Divideix la fila i pel pivot
        double pivot = A[i * ldA + i];
        if (pivot == 0.0)
        {
            fprintf(stderr, "[ERROR] Gauss-Jordan: pivot nul a la fila %d\n", i);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < n; j++)
        {
            // A és stride=ldA, A_inv és contigua => i*n + j
            A[i * ldA + j] /= pivot;
            A_inv[i * n + j] /= pivot;
        }

        // 2) Anul·la la columna i per a totes les altres files
        for (int fila = 0; fila < n; fila++)
        {
            if (fila != i)
            {
                double pivot2 = A[fila * ldA + i];
                for (int k = 0; k < n; k++)
                {
                    // Resta a la fila 'fila' la fila 'i' multiplicada per pivot2
                    A[fila * ldA + k] -= A[i * ldA + k] * pivot2;
                    A_inv[fila * n + k] -= A_inv[i * n + k] * pivot2;
                }
            }
        }
    }
}
void schur_inverse(double *A, int ldA,
                   double *A_inv, int ldInv,
                   int n, int base_case_size)
{
    // ----- CAS BASE -----
    if (n <= base_case_size)
    {
        if (n == 1)
        {
            // Base case: inverse of 1x1 matrix
            A_inv[0] = 1.0 / A[0];
            return;
        }
        if (n == 2)
        {
            inverse_2x2_stride(A, ldA, A_inv, ldInv);
            return;
        }
        GJElimination_stride(A, ldA, A_inv, n);
        return;
    }

    // ----- PARTICIO -----
    int half = n / 2;

    // Subblocs (cadascun de mida half x half)
    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + ldA * half;
    double *A22 = A + ldA * half + half;

    double *A11_inv = A_inv;
    double *A12_inv = A_inv + half;
    double *A21_inv = A_inv + ldInv * half;
    double *A22_inv = A_inv + ldInv * half + half;

    // ----- RESERVA TEMPORALS -----
    double *A11_inv_temp = (double*)malloc(half * half * sizeof(double));
    double *temp1        = (double*)malloc(half * half * sizeof(double));
    double *temp2        = (double*)malloc(half * half * sizeof(double)); // A11_inv_temp * A12
    double *temp3        = (double*)malloc(half * half * sizeof(double)); // Per a S, etc.
    double *S_inv        = (double*)malloc(half * half * sizeof(double));
    memset(temp1, 0, half * half * sizeof(double));
    memset(temp2, 0, half * half * sizeof(double));
    memset(temp3, 0, half * half * sizeof(double));
    memset(temp1, 0, half * half * sizeof(double));


    // ----- 1) Invertim A11 recursivament -----
    schur_inverse(A11, ldA,A11_inv_temp, half, half, base_case_size);

    // ----- 2) Calcul de temp2 = A11_inv_temp * A12 -----
    matmul_blocked(A11_inv_temp, half, A12, ldA, temp2, half, half, half, half);

    // ----- 3) Calcul de temp1 = A21 * temp2 -----
    matmul_blocked(A21, ldA, temp2, half, temp1, half, half, half, half);

    // ----- 4) Calcul de S = A22 - temp1 -----
    //     guardem S a temp3 
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp3[i * half + j] = A22[i * ldA + j] - temp1[i * half + j];
        }
    }

    // ----- 5) Invertim S recursivament: S_inv -----
    schur_inverse(temp3, half, S_inv, half, half, base_case_size);
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A22_inv[i * ldInv + j] = S_inv[i * half + j];
        }
    }

    // ---------------------
    // ========== BLOCS PER AL CÀLCUL FINAL ==========
    //
    // A12_inv = -( (A11_inv_temp*A12)*S_inv ) = -(temp2 * S_inv).

    // 6) temp3 = temp2 * S_inv
    memset(temp3, 0, half * half * sizeof(double));
    matmul_blocked(temp2, half, S_inv, half, temp3, half, half, half, half);
    // 7) A12_inv = -temp3
    //Es podria millorar aixo incrustant-ho dins del matmul_blocked de amunt
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A12_inv[i * ldInv + j] = -1.0 * temp3[i * half + j];
        }
    }
           /*
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < half; k++)
            {
                sum += temp2[i * half + k] * S_inv[k * half + j];
            }
            temp3[i * half + j] = sum;
            A12_inv[i * ldInv + j] = -1 * sum; //calcul A12_inv incrustat
        }
    }
*/
    // ---------------------
    // ========== A11_inv = A11_inv_temp + (temp2*S_inv*A21)*A11_inv_temp ==========
    //
    // Ja tenim temp3 = temp2*S_inv, així que primer farem:
    //    temp1 = temp3 * A21
    // després:
    //    (temp1 * A11_inv_temp)
    // i finalment sumem a A11_inv_temp.
    // ---------------------

    // 8) temp1 = temp3 * A21 
    memset(temp1, 0, half * half * sizeof(double));
    matmul_blocked(temp3, half, A21, ldA, temp1, half, half, half, half);

    // 9) Ara multipliquem temp1 * A11_inv_temp => ho deixarem altre cop a temp3
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < half; k++)
            {
                sum += temp1[i * half + k] * A11_inv_temp[k * half + j];
            }
            //S[i * half + j] = sum;
            A11_inv[i * ldInv + j] = A11_inv_temp[i * half + j] + sum;
        }
    }

    // ---------------------
    // ========== A21_inv = -( S_inv * A21 * A11_inv_temp ) ==========
    //
    //  farem:
    //    temp1 = S_inv * A21
    //    temp3 = temp1 * A11_inv_temp
    //    A21_inv = -temp3
    // ---------------------

    // 11) temp1 = S_inv * A21
    memset(temp1, 0, half * half * sizeof(double));
    matmul_blocked(S_inv, half, A21,  ldA, temp1, half, half, half, half);

    // 12) Ara temp3 = temp1 * A11_inv_temp
    memset(temp3, 0, half * half * sizeof(double));
    matmul_blocked(temp1, half, A11_inv_temp, half, temp3, half, half, half, half);
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A21_inv[i * ldInv + j] = - temp3[i * half + j];
        }
    }

    // ---------------------
    // ========== A22_inv = S_inv ==========
    // ---------------------

    // ----- ALLIBERAR MEMÒRIA TEMPORAL -----
    free(A11_inv_temp);
    free(temp1);
    free(temp2);
    free(temp3);
    free(S_inv);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <matrix_size> <base_case_size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    int base_case_size = atoi(argv[2]);

    if (N <= 0 || base_case_size <= 0)
    {
        printf("Error: Ensure matrix_size > 0, base_case_size > 0, and matrix_size is divisible by base_case_size.\n");
        return 1;
    }
    double *A = (double *)malloc(N * N * sizeof(double));
    double *A_inv = (double *)malloc(N * N * sizeof(double));
    memset(A_inv, 0, N * N * sizeof(double));
    srand(0u);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = ((double)rand() / RAND_MAX);
    printf("Print A:\n");
    // print_matrix(A);
    schur_inverse(A, N, A_inv, N, N, base_case_size);
    // print_matrix(A_inv);
    double sum = 0.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += A_inv[i * N + j];
    free(A);
    free(A_inv);
    printf("Checksum A:%.8f \n", sum);

    return 0;
}
