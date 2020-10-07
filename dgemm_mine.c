#include <immintrin.h>
#include <stdio.h>
#include <time.h>

const char* dgemm_desc = "An implementation of GotoBlas";

const int kc = 8;
const int mc = 1024;
int aa = 0;

void square_dgemm(const int M, double *A, double *B, double *C)
{
    int i, i1, j, j1, k, k1, KC, MC;
    register __m256d c;
    register __m256d b;
    __m256d a[kc];
    __m256d *p_a;
    double *p_B;

    for (k = 0; k < M; k += kc) {
        KC = M - k < kc ? M - k : kc;
        for (i = 0; i < M; i += mc) {
            MC = M - i < mc ? M - i : mc;
            for (j = 0; j < M; j += 4)
                for (i1 = i; i1 < i + MC; i1 += 4) {
                    for (k1 = 0; k1 < kc; k1++)
                        a[k1] = _mm256_loadu_pd(A + (k + k1) * M + i1);
                    if (i1 + 4 <= i + MC) {
                        for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                            c = _mm256_loadu_pd(C + j1 * M + i1);
                            for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                                b = _mm256_set1_pd(*(p_B++));
                                c = _mm256_fmadd_pd(*(p_a++), b, c);
                                aa++;
                            }
                            _mm256_storeu_pd(C + j1 * M + i1, c);
                        }
                    }
                    else if (i1 + 3 == i + MC) {
                        for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                            c = _mm256_loadu_pd(C + j1 * M + i1);
                            for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                                b = _mm256_set1_pd(*(p_B++));
                                c = _mm256_fmadd_pd(*(p_a++), b, c);
                                aa++;
                            }
                            *(C + j1 * M + i1) = c[0];
                            *(C + j1 * M + i1 + 1) = c[1];
                            *(C + j1 * M + i1 + 2) = c[2];
                        }
                    }
                    else if (i1 + 2 == i + MC) {
                        for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                            c = _mm256_loadu_pd(C + j1 * M + i1);
                            for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                                b = _mm256_set1_pd(*(p_B++));
                                c = _mm256_fmadd_pd(*(p_a++), b, c);
                                aa++;
                            }
                            *(C + j1 * M + i1) = c[0];
                            *(C + j1 * M + i1 + 1) = c[1];
                        }
                    }
                    else {
                        for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                            c = _mm256_loadu_pd(C + j1 * M + i1);
                            for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                                b = _mm256_set1_pd(*(p_B++));
                                c = _mm256_fmadd_pd(*(p_a++), b, c);
                                aa++;
                            }
                            *(C + j1 * M + i1) = c[0];
                        }
                    }

                    /*
                    for (k1 = 0; k1 < KC; k1++)
                        a[k1] = _mm256_loadu_pd(A + (k + k1) * M + i1);
                    for (j1 = j; j1 < j + 4 && j1 < M; j1++) {
                        c = _mm256_loadu_pd(C + j1 * M + i1);
                        for (k1 = 0, p_B = B + j1 * M + k + k1, p_a = a; k1 < KC; k1++) {
                            b = _mm256_set1_pd(*(p_B++));
                            c = _mm256_fmadd_pd(*(p_a++), b, c);
                            aa++;
                        }
                        if (i1 + 4 <= i + MC)
                            _mm256_storeu_pd(C + j1 * M + i1, c);
                        else if (i1 + 3 == i + MC) {
                            *(C + j1 * M + i1) = c[0];
                            *(C + j1 * M + i1 + 1) = c[1];
                            *(C + j1 * M + i1 + 2) = c[2];
                        }
                        else if (i1 + 2 == i + MC) {
                            *(C + j1 * M + i1) = c[0];
                            *(C + j1 * M + i1 + 1) = c[1];
                        }
                        else
                            *(C + j1 * M + i1) = c[0];
                    }
                    */
                }
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
    printf("%d\n", aa);

    return 0;
}


