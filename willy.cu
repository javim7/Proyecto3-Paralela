#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declarar las variables en memoria constante.
_constant_ float d_Cos[degreeBins];
_constant_ float d_Sin[degreeBins];

// Función para calcular el tiempo transcurrido entre dos eventos CUDA.
float GetElapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, sizeof(int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
    {
      int idx = j * w + i;
      if (pic[idx] > 0)
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;
        float theta = 0;
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++;
          theta += radInc;
        }
      }
    }
}

// Kernel memoria compartida
_global_ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  _shared_ int sharedAcc[rBins * degreeBins];
  sharedAcc[threadIdx.x] = 0;
  __syncthreads();

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd(&sharedAcc[rIdx * degreeBins + tIdx], 1);
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < rBins * degreeBins; i += blockDim.x)
  {
    atomicAdd(&acc[i], sharedAcc[i]);
  }
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);

    // Crear eventos CUDA para medir el tiempo.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256);

    // Registrar el tiempo de inicio.
    cudaEventRecord(start);

    GPU_HoughTranShared<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Registrar el tiempo de finalización.
    cudaEventRecord(stop);

    // Sincronizar para asegurarse de que el kernel haya terminado.
    cudaDeviceSynchronize();

    // Calcular el tiempo transcurrido.
    float elapsedTime = GetElapsedTime(start, stop);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    for (int i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");

    printf("Tiempo transcurrido: %f ms\n", elapsedTime);

    // Limpieza
    free(pcCos);
    free(pcSin);
    delete[] cpuht;
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);
    
    // Destruir los eventos CUDA.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}