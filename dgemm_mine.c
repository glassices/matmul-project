#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const char* dgemm_desc = "This project is too hard.";

#if !defined(BLOCK_SIZE)
  #define BLOCK_SIZE 32
#endif

#define min(a,b) (((a) < (b))? (a) : (b))

void *aligned_malloc(size_t align, size_t size) {
    void *mem = malloc(size+align+sizeof(void*));
    void **ptr = (void**)((uintptr_t)(mem+align+sizeof(void*)) & ~(align-1));
    ptr[-1] = mem;
    return ptr;
}

void aligned_free(void *ptr) {
    free(((void**)ptr)[-1]);
}

void do_block(int lda, int K, double* restrict A, double* restrict B, double* restrict C)
{
    __builtin_assume_aligned(A, 32);
    __builtin_assume_aligned(B, 32);
    __builtin_assume_aligned(C, 32);

    int i, j, k;
    __m256d a0, a4, b, c0_0, c0_1, c0_2, c0_3, c4_0, c4_1, c4_2, c4_3;
    double *B_kj, *C_ij;
    for (j = 0; j < BLOCK_SIZE; j += 4)
        for (i = 0; i < BLOCK_SIZE; i += 8) {
            C_ij = C + j * lda + i;
            c0_0 = _mm256_load_pd(C_ij);
            c4_0 = _mm256_load_pd(C_ij + 4);
            c0_1 = _mm256_load_pd(C_ij + lda);
            c4_1 = _mm256_load_pd(C_ij + lda + 4);
            c0_2 = _mm256_load_pd(C_ij + lda * 2);
            c4_2 = _mm256_load_pd(C_ij + lda * 2 + 4);
            c0_3 = _mm256_load_pd(C_ij + lda * 3);
            c4_3 = _mm256_load_pd(C_ij + lda * 3 + 4);
            for (k = 0; k < K; k++) {
                a0 = _mm256_load_pd(A + k * lda + i);
                a4 = _mm256_load_pd(A + k * lda + i + 4);
                B_kj = B + j * lda + k;
                b = _mm256_set1_pd(*B_kj);
                c0_0 = _mm256_fmadd_pd(a0, b, c0_0);
                c4_0 = _mm256_fmadd_pd(a4, b, c4_0);
                b = _mm256_set1_pd(*(B_kj + lda));
                c0_1 = _mm256_fmadd_pd(a0, b, c0_1);
                c4_1 = _mm256_fmadd_pd(a4, b, c4_1);
                b = _mm256_set1_pd(*(B_kj + lda * 2));
                c0_2 = _mm256_fmadd_pd(a0, b, c0_2);
                c4_2 = _mm256_fmadd_pd(a4, b, c4_2);
                b = _mm256_set1_pd(*(B_kj + lda * 3));
                c0_3 = _mm256_fmadd_pd(a0, b, c0_3);
                c4_3 = _mm256_fmadd_pd(a4, b, c4_3);
            }
            _mm256_store_pd(C_ij, c0_0);
            _mm256_store_pd(C_ij + 4, c4_0);
            _mm256_store_pd(C_ij + lda, c0_1);
            _mm256_store_pd(C_ij + lda + 4, c4_1);
            _mm256_store_pd(C_ij + lda * 2, c0_2);
            _mm256_store_pd(C_ij + lda * 2 + 4, c4_2);
            _mm256_store_pd(C_ij + lda * 3, c0_3);
            _mm256_store_pd(C_ij + lda * 3 + 4, c4_3);
        }
}


void square_dgemm (const int raw_lda, const double* rawA, const double* rawB, double* rawC)
{
    clock_t start;

    //start = clock();
    int lda = ((raw_lda - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
    double *A = aligned_malloc(BLOCK_SIZE, sizeof(double) * lda * lda * 3);
    double *B = A + (size_t)lda * lda;
    double *C = B + (size_t)lda * lda;
    for (int i = 0; i < raw_lda; i++) {
        memcpy(A + i * lda, rawA + i * raw_lda, sizeof(double) * raw_lda);
        memcpy(B + i * lda, rawB + i * raw_lda, sizeof(double) * raw_lda);
        memcpy(C + i * lda, rawC + i * raw_lda, sizeof(double) * raw_lda);
    }
    //printf("%u\n", clock() - start);


    //start = clock();
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            for (int k = 0; k < lda; k += BLOCK_SIZE)
                do_block(lda, min(BLOCK_SIZE, raw_lda - k), A + i + k*lda, B + k + j*lda, C + i + j*lda);
    //printf("%u\n", clock() - start);

        
    //start = clock();

    for (int j = 0; j < raw_lda; j++)
        memcpy(rawC + j * raw_lda, C + j * lda, sizeof(double) * raw_lda);

    //printf("%u\n", clock() - start);
    aligned_free(A);
}

/*
int main()
{
    int n = 1000;
    double *A = malloc(sizeof(double) * 1400 * 1400);
    double *B = malloc(sizeof(double) * 1400 * 1400);
    double *C = malloc(sizeof(double) * 1400 * 1400);
    square_dgemm(n, A, B, C);
    free(A);
    free(B);
    free(C);

    return 0;
}
*/
