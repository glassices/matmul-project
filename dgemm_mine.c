#include <immintrin.h>
#include <stdio.h>
#include <time.h>

const char* dgemm_desc = "An implementation of GotoBlas";

const int kc = 16;
const int step_j = 8;

void square_dgemm(const int M, double *A, double *B, double *C)
{
    int i, j, j1, k, k1, KC, MC;
    register __m256d c;
    register __m256d b;
    __m256d a[kc];
    __m256d *p_a;
    double *p_B;

    for (k = 0; k < M; k += kc) {
        KC = M - k < kc ? M - k : kc;
        for (j = 0; j < M; j += step_j) {
            for (i = 0; i < M / 4 * 4; i += 4) {
                for (k1 = 0; k1 < kc; k1++)
                    a[k1] = _mm256_loadu_pd(A + (k + k1) * M + i);
                for (j1 = j; j1 < j + step_j && j1 < M; j1++) {
                    c = _mm256_loadu_pd(C + j1 * M + i);
                    for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                        b = _mm256_set1_pd(*(p_B++));
                        c = _mm256_fmadd_pd(*(p_a++), b, c);
                    }
                    _mm256_storeu_pd(C + j1 * M + i, c);
                }
            }
            /*
            if (M % 4) {
                i = M / 4 * 4;
                for (k1 = 0; k1 < kc; k1++)
                    a[k1] = _mm256_loadu_pd(A + (k + k1) * M + i);
                for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                    c = _mm256_loadu_pd(C + j1 * M + i);
                    for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                        b = _mm256_set1_pd(*(p_B++));
                        c = _mm256_fmadd_pd(*(p_a++), b, c);
                    }
                    if (i + 3 == M) {
                        *(C + j1 * M + i) = c[0];
                        *(C + j1 * M + i + 1) = c[1];
                        *(C + j1 * M + i + 2) = c[2];
                    }
                    else if (i + 2 == M) {
                        *(C + j1 * M + i) = c[0];
                        *(C + j1 * M + i + 1) = c[1];
                    }
                    else {
                        *(C + j1 * M + i) = c[0];
                    }
                }
            }
            */
        }
    }
}

int main()
{
    clock_t start;
    start = clock();


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


    int n = 1500;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    square_dgemm(n, A, B, C);
    free(A);
    free(B);
    free(C);
    printf("%u\n", clock() - start);

    return 0;
}

