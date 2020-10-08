#include <immintrin.h>
#include <stdio.h>
#include <time.h>

const char* dgemm_desc = "An implementation of GotoBlas";

const int kc = 7;
const int step_j = 16;

void square_dgemm(const int M, double *A, double *B, double *C)
{
    int i, j, j1, k, k1, KC, MC;
    register __m256d c0, c1;
    register __m256d b;

    __m256d a0[kc], a1[kc];

    for (j = 0; j < M; j += step_j) {
        for (k = 0; k < M; k += kc) {
            KC = M - k < kc ? M - k : kc;
            for (i = 0; i < M / 8 * 8; i += 8) {
                for (k1 = 0; k1 < kc; k1++) {
                    a0[k1] = _mm256_loadu_pd(A + (k + k1) * M + i);
                    a1[k1] = _mm256_loadu_pd(A + (k + k1) * M + i + 4);
                }
                for (j1 = j; j1 < j + step_j && j1 < M; j1++) {
                    c0 = _mm256_loadu_pd(C + j1 * M + i);
                    c1 = _mm256_loadu_pd(C + j1 * M + i + 4);
                    for (k1 = 0; k1 < KC; k1++) {
                        b = _mm256_set1_pd(*(B + j1 * M + k + k1));
                        c0 = _mm256_fmadd_pd(a0[k1], b, c0);
                        c1 = _mm256_fmadd_pd(a1[k1], b, c1);
                    }
                    _mm256_storeu_pd(C + j1 * M + i, c0);
                    _mm256_storeu_pd(C + j1 * M + i + 4, c1);
                }
            }
            if (M % 8) {
                i = M / 8 * 8;
                for (k1 = 0; k1 < kc; k1++) {
                    a0[k1] = _mm256_loadu_pd(A + (k + k1) * M + i);
                    a1[k1] = _mm256_loadu_pd(A + (k + k1) * M + i + 4);
                }
                for (j1 = j; j1 < j + step_j && j1 < M; j1++) {
                    c0 = _mm256_loadu_pd(C + j1 * M + i);
                    c1 = _mm256_loadu_pd(C + j1 * M + i + 4);
                    for (k1 = 0; k1 < KC; k1++) {
                        b = _mm256_set1_pd(*(B + j1 * M + k + k1));
                        c0 = _mm256_fmadd_pd(a0[k1], b, c0);
                        c1 = _mm256_fmadd_pd(a1[k1], b, c1);
                    }
                    for (k1 = 0; k1 < 4; k1++)
                        if (i + k1 < M)
                            *(C + j1 * M + i + k1) = c0[k1];
                    for (k1 = 0; k1 < 4; k1++)
                        if (i + k1 + 4 < M)
                            *(C + j1 * M + i + k1 + 4) = c1[k1];
                }
            }
        }
    }
}

#if 0
int main()
{
    clock_t start;


    /*
    register __m256d a, b, c;
    double sum = 0;
    int i, j;
    for (i = 0; i < 843750000; i++) {
        b = _mm256_set1_pd(1.0);
        c = _mm256_fmadd_pd(a, b, c);
    }
    printf("%lf\n", c[0]);
    printf("%u\n", clock() - start);
    */

    /*
    register double a, b, c;
    for (int i = 0; i < 843750000; i++)
        c = a * b + c;
    printf("%lf\n", c);
    printf("%u\n", clock() - start);
    */


    int n = 1600;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    start = clock();
    square_dgemm(n, A, B, C);
    printf("%u\n", clock() - start);
    free(A);
    free(B);
    free(C);

    return 0;
}

#endif
