#include <cstdio>
#include <cstdlib>
using namespace std;

/**
 * @brief adição de vetor na GPU
 * 
 * @param a vetor a ser somado
 * @param b vetor a ser somado
 * @param c vetor resultante
 * @param n número máximo de elementos nos vetores
 */
__global__ void add(int* a, int* b, int* c, int n) {
	// calcula o ID de cada thread
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// verifica se está dentro do intervalo do vetor
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}

int main() {
	// define o numero máximo de elementos
	const int N = 1 << 20;
	size_t bytes = sizeof(int) * N;

	// prepara os vetores
	int *a, *b, *c;

	// gerencia a memória na GPU
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// preenche os vetores com números aleatórios
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	int threads = 64;
	int blocks = (int)ceil(N / threads);

	// executa na GPU
	add << <blocks, threads >> > (a, b, c, N);

	// espera GPU executar e sincroniza a memória
	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) {
		printf("%3d + %3d = %3d\n", a[i], b[i], c[i]);
	}

	return 0;
}