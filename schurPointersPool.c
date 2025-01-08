#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int N = 4;
static double *g_pool = NULL;     // la memòria base
static size_t  g_pool_size = 0;   // mida total (en doubles o bytes)
static size_t  g_pool_offset = 0; // offset actual (índex)


void* pool_alloc(size_t num_doubles)
{
    // Mirem si hi ha espai suficient
    if (g_pool_offset + num_doubles > g_pool_size) {
        fprintf(stderr, "pool_alloc: Out of memory pool!\n");
        exit(EXIT_FAILURE);
    }
    void *ptr = &g_pool[g_pool_offset];
    g_pool_offset += num_doubles;
    return ptr;
}

// "pop" tipus stack (desfem l'última reserva)
void pool_free(size_t num_doubles)
{
    // Assumim que alliberem exactament en ordre invers
    g_pool_offset -= num_doubles;
}

size_t calculate_pool_size(int N, int base_case_size) {
    size_t total_memory = 0;
    int current_size = N;

    while (current_size > base_case_size) {
        // Cada nivell recursiu necessita memòria per a 5 matrius temporals
        total_memory += 5 * (current_size / 2) * (current_size / 2);
        current_size /= 2;
    }

    // Afegim una mica de memòria addicional per seguretat
    total_memory += 10 * base_case_size * base_case_size;

    return total_memory;
}

void print_matrix(double *A)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%8.4f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_rec(double *A, int size)
{
    for (int i = 0; i < size; i++)
    {   
        for (int j = 0; j < size; j++)
        {
            printf("%8.4f ", A[j + N * i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_size(double *A, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%8.4f ", A[j + size * i]);
        }
        printf("\n");
    }
    printf("\n");
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
void schur_inverse(double *A, int ldA, double *A_inv, int ldInv, int n, int base_case_size)
{
    if (n <= base_case_size)
    {
        if (n == 1)
        {
            // Base case: inverse of 1x1 matrix
            A_inv[0] = 1 / A[0];
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

    int half = n / 2;

    // Allocate memory for blocks
    // Abans utilitzavem n, ara ldA
    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + ldA * half;
    double *A22 = A + ldA * half + half;

    double *A11_inv = A_inv;
    double *A12_inv = A_inv + half;
    double *A21_inv = A_inv + ldInv * half;
    double *A22_inv = A_inv + ldInv * half + half;

    //size_t offset_inicial = g_pool_offset;
    
    double *S = (double*) pool_alloc(half * half);
    double *S_inv = (double*) pool_alloc(half * half);
    double *A11_inv_temp = (double*) pool_alloc(half * half);

    // Recursive call for A11_inv
    schur_inverse(A11, ldA, A11_inv_temp, half, half, base_case_size);
    // printf("A11inv:\n");
    // print_matrix_size(A11_inv_temp, half);

    // Compute Schur complement S = A22 - A21 * A11_inv * A12
    double *temp1 = (double*) pool_alloc(half * half);
    double *temp2 = (double*) pool_alloc(half * half);

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp1[i * half + j] = 0.0;
            for (int k = 0; k < half; k++)
            {
                // printf("A21[%d,%d] = %f\n", i, j, A21[i * ldA + j]);
                // printf("A11_inv[%d,%d] = %f\n", k, j, A11_inv_temp[k * half + j]);
                temp1[i * half + j] += A21[i * ldA + k] * A11_inv_temp[k * half + j];
            }
        }
    }
    // tmp1*A12 = temp2
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp2[i * half + j] = 0.0;
            for (int k = 0; k < half; k++)
            {
                // printf("temp1[%d,%d] = %f\n", i, k, temp1[i * half + k]);
                // printf("A12[%d,%d] = %f\n", k, j, A12[k * ldA + j]);
                temp2[i * half + j] += temp1[i * half + k] * A12[k * ldA + j];
            }
        }
    }
    // S = A22 - temp2
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            // printf("A22[%d,%d] = %f\n", i, j, A22[i * ldA + j]);
            // printf("temp2[%d,%d] = %f\n", i, j, temp2[i * half + j]);

            S[i * half + j] = A22[i * ldA + j] - temp2[i * half + j];
        }
    }

    // Recursive call for S_inv
    schur_inverse(S, half, S_inv, half, half, base_case_size);
    // printf("S_inv:\n");
    // print_matrix_size(S_inv, half);

    // A_inv11 = A11_inv + A11_inv * A12 * S_inv * A21 * A11_inv
    // Ordre de les operacions:
    // temp1 = A11_inv * A12
    // temp2 = temp1 * S_inv
    // temp1 = temp2 * A21
    // A_inv11 = temp1 * A11_inv
    // A_inv11 += A11_inv

    // matmul(A11_inv, A12, temp1, half);
    // A11_inv * A12 = temp1
    // A11_inv tamany N, A12 tamany N, temp1 tamany half
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp1[i * half + j] = 0;
            for (int k = 0; k < half; k++)
            {
                temp1[i * half + j] += A11_inv_temp[i * half + k] * A12[k * ldA + j];
            }
        }
    }

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp2[i * half + j] = 0;
            for (int k = 0; k < half; k++)
            {
                temp2[i * half + j] += temp1[i * half + k] * S_inv[k * half + j];
            }
        }
    }

    // matmul(temp2, A21, temp1, half);
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp1[i * half + j] = 0;
            for (int k = 0; k < half; k++)
            {
                temp1[i * half + j] += temp2[i * half + k] * A21[k * ldA + j];
            }
        }
    }

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp2[i * half + j] = 0;
            for (int k = 0; k < half; k++)
            {
                temp2[i * half + j] += temp1[i * half + k] * A11_inv_temp[k * half + j];
            }
        }
    }
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A11_inv[i * ldInv + j] = A11_inv_temp[i * half + j] + temp2[i * half + j];
        }
    }

    // printf("A_inv11:\n");
    // print_matrix_rec(A11_inv, half);

    // A12_inv = -A11_inv_temp * A12 * S_inv
    // matmul(A11_inv, A12, temp1, half);

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp1[i * half + j] = 0.0;
            for (int k = 0; k < half; k++)
            {
                temp1[i * half + j] += A11_inv_temp[i * half + k] * A12[k * ldA + j];
            }
        }
    }
    // A12_inv = - temp1 * S_inv
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            // Usem una variable local per acumular
            double accum = 0.0;
            for (int k = 0; k < half; k++)
            {
                // temp1 té stride = half
                // S_inv té stride = half
                accum += temp1[i * half + k] * S_inv[k * half + j];
            }
            // Apliquem el signe negatiu
            accum = -accum;

            // Ara fem UN SOL store en A12_inv
            // (incrustada a la matriu gran de dimensió N)
            A12_inv[i * ldInv + j] = accum;
        }
    }
    // printf("A12_inv:\n");
    // print_matrix_rec(A12_inv, half);

    // A21_inv = -S_inv * A21 * A11_inv_temp

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            temp1[i * half + j] = 0.0;
            for (int k = 0; k < half; k++)
            {
                temp1[i * half + j] += S_inv[i * half + k] * A21[k * ldA + j];
            }
        }
    }

    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            double accum = 0.0;
            for (int k = 0; k < half; k++)
            {
                accum += temp1[i * half + k] * A11_inv_temp[k * half + j];
            }
            accum = -accum;
            A21_inv[i * ldInv + j] = accum;
        }
    }
    // printf("A_inv21:\n");
    // print_matrix_rec(A21_inv, half);

    // A_inv22 = S_inv
    for (int i = 0; i < half; i++)
    {
        for (int j = 0; j < half; j++)
        {
            A22_inv[i * ldInv + j] = S_inv[i * half + j];
        }
    }
    pool_free(half * half);  // temp2
    pool_free(half * half);  // temp1
    pool_free(half * half);  // A11_inv_temp
    pool_free(half * half);  // S_inv
    pool_free(half * half);  // S

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

    if (N <= 0 || base_case_size <= 0) {
        printf("Error: Ensure matrix_size > 0, base_case_size > 0.\n");
        return 1;
    }
    size_t pool_size = calculate_pool_size(N, base_case_size);
    printf("Calculated pool size: %zu doubles\n", pool_size);
    g_pool_size = pool_size;
    if (posix_memalign((void**)&g_pool, 64, g_pool_size * sizeof(double)) != 0) {
        fprintf(stderr, "Error allocating aligned memory.\n");
        exit(EXIT_FAILURE);
    }

    /*
    if (posix_memalign((void**)&A, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "Error allocating aligned memory for A.\n");
        exit(EXIT_FAILURE);
    }

    if (posix_memalign((void**)&A_inv, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "Error allocating aligned memory for A_inv.\n");
        free(A); // Alliberem la memòria de A si falla
        exit(EXIT_FAILURE);
    }
*/
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
    free(g_pool);
    printf("Checksum A:%.8f \n", sum);

    return 0;
}
