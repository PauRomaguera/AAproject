#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int N = 4;
#include <stddef.h> // per a size_t
#define BS 16  // Bloc de 64x64
#define ALIGN_BYTES 64  // 64 bytes per alinear a AVX, per exemple
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

void GJElimination_stride_blocked(double *A, int ldA, double *A_inv, int n)
{
    // 0) Inicialitzar A_inv com a identitat contigua (n×n)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_inv[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // 1) Gauss-Jordan
    for (int i = 0; i < n; i++)
    {
        // --- PAS 1: Escalament de la fila pivot 'i' ---
        double pivot = A[i * ldA + i];
        if (pivot == 0.0) {
            fprintf(stderr, "[ERROR] Gauss-Jordan: pivot nul a la fila %d\n", i);
            exit(EXIT_FAILURE);
        }

        // Dividim la fila A i la fila A_inv per pivot
        // -> (A[i,0..n-1]) /= pivot
        // -> (A_inv[i,0..n-1]) /= pivot
        {
            // Fent-ho d'una sola tirada, sense blocking, perquè és una sola fila
            for (int col = 0; col < n; col++) {
                A[i * ldA + col] /= pivot;
                A_inv[i*n + col] /= pivot;
            }
        }

        // --- PAS 2: Anul·lar la columna i en totes les altres files ---
        //    Aquí introduïm "blocking" en l'eix de columnes (k).
        for (int fila = 0; fila < n; fila++)
        {
            if (fila == i) continue;  // no eliminem la fila pivot

            double factor = A[fila * ldA + i];
            if (factor != 0.0)
            {
                // En lloc de fer directament un bucle col=0..n,
                // ho fem en blocs de columns de mida BS
                for (int colBlock = 0; colBlock < n; colBlock += BS) {
                    int colEnd = (colBlock + BS < n) ? (colBlock + BS) : n;

                    // Ara fem la subtract en el rang colBlock..colEnd
                    for (int col = colBlock; col < colEnd; col++) {
                        A[fila * ldA + col] -= A[i * ldA + col] * factor;
                        A_inv[fila*n + col] -= A_inv[i*n + col] * factor;
                    }
                }
            }
        }
    }
}

//Resta i store d exemple
/*
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp3[i * half + j] = A22[i * ldA + j] - temp1[i * half + j];
        }
    }

    schur_inverse(temp3, half, S_inv, half, half, base_case_size);
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A22_inv[i * ldInv + j] = S_inv[i * half + j];
        }
    }
*/
void schur_inverse(double *A, int ldA,
                   double *A_inv, int ldInv,
                   int n, int base_case_size)
{
    if (n <= base_case_size) {
        GJElimination_stride(A, ldA, A_inv, n);
        return;
    }

    int half = n / 2;

    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + ldA * half;
    double *A22 = A + ldA * half + half;

    double *A11_inv = A_inv;
    double *A12_inv = A_inv + half;
    double *A21_inv = A_inv + ldInv * half;
    double *A22_inv = A_inv + ldInv * half + half;

    size_t blockSize = (size_t)half * (size_t)half * sizeof(double);

    double *R1 = NULL;
    double *R2 = NULL;
    double *R3 = NULL;
    double *R4 = NULL;
    double *R5 = NULL;

    // --- Crides a posix_memalign totes seguides (sense if) ---
    posix_memalign((void **)&R1, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R2, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R3, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R4, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R5, ALIGN_BYTES, blockSize);
    //cal fer memset a 0 cada cop que es faci store a una matriu temporal
    memset(R1, 0, half * half * sizeof(double));
    memset(R2, 0, half * half * sizeof(double));
    memset(R3, 0, half * half * sizeof(double));
    memset(R4, 0, half * half * sizeof(double));
    memset(R5, 0, half * half * sizeof(double));


    //R1 = A11_inv
    schur_inverse(A11, ldA, R1, half, half, base_case_size);

    //R2 = A21 * R1 (Depends on R1)
    matmul_blocked(A21, ldA, R1, half, R2, half, half, half, half);

    //R3 = R1 * A12 (Depends on R1)
    matmul_blocked(R1, half, A12, ldA, R3, half, half, half, half);

    //R4 = A21*R3 (Depends on R3)
    matmul_blocked(A21, ldA, R3, half, R4, half, half, half, half);

    //R5 = R4 - A22 (Depends on R4)
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            R5[i * half + j] = A22[i * ldA + j] - R4[i * half + j];
        }
    }
    //R4 = R5_inv (S_inv) (Depen de R5)
    memset(R4, 0, half * half * sizeof(double));
    schur_inverse(R5, half, R4, half, half, base_case_size);

    //A22_inv = R4 
    //Probar amb memcopy
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A22_inv[i * ldInv + j] = R4[i * half + j];
        }
    }

    //A12_inv = R3*R4 (Probar si es pot fer junt amb el calcul de R3*A21_inv) (Depen de R4)
    matmul_blocked(R3, half, R4, half, A12_inv, ldInv, half, half, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A12_inv[i*ldInv + j] = - A12_inv[i*ldInv + j];
        }
    }
    //A21_inv = R4*R2 (Depen de R4)
    matmul_blocked(R4, half, R2, half, A21_inv, ldInv, half, half, half);
    //R5 = R3 * A21_inv
    memset(R5, 0, half * half * sizeof(double));
    matmul_blocked(R3, half, A21_inv, ldInv, R5, half, half, half, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A21_inv[i*ldInv + j] = - A21_inv[i*ldInv + j];
        }
    }
    //A11_inv = R1 + R5
    //matmul_blocked(R1, half, R5, half, A11_inv, ldInv, half, half, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11_inv[i * ldInv + j] = R1[i * half + j] + R5[i * half + j];
        }
    }

    free(R1);
    free(R2);
    free(R3);
    free(R4);
    free(R5);
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
