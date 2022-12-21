#include <cstdlib>
#include <cstdio>
using namespace std;

/**
 * @brief adição de vetor na GPU
 *
 * @param a vetor a ser somado
 * @param b vetor a ser somado
 * @param c vetor resultante
 * @param n número máximo de elementos nos vetores
 */
__global__ void add(int* a, int* b, int* c, unsigned n) {
	// calcula o ID de cada thread
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// verifica se está dentro do intervalo do vetor
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}

int main() {
	// define o numero máximo de elementos
	const int N = 1 << 16;

	printf("N: %u\n", N);

	// prepara os vetores
	int a[N], b[N], c[N];

	// preenche os vetores com números aleatórios
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	printf("N %d\n", N);
	printf("sizeof(int) %d\n", sizeof(int));
	printf("sizeof(a[N]) %d\n", sizeof(a));

	// gerencia a memória na GPU
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, sizeof(a));
	cudaMalloc(&d_b, sizeof(b));
	cudaMalloc(&d_c, sizeof(c));

	 // copia dados para a GPU
	cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

	int threads = 64;
	int blocks = (int)ceil(N / threads);

	// executa na GPU
	add<<<blocks, threads>>>(d_a, d_b, d_c, N);

	// espera GPU executar e sincroniza a memória
	cudaMemcpy(c, d_c, sizeof(c), cudaMemcpyDeviceToHost);

	// limpa a memória na GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	for (int i = 0; i < N; i++) {
		//printf("%3d + %3d = %3d\n", a[i], b[i], c[i]);
	}

	return 0;
}