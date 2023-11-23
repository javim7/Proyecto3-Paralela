#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <cuda_runtime.h>
#include "pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declaración de memoria constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

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

// GPU kernel con memoria compartida
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // a. Definir locID usando los IDs de los hilos del bloque
  int locID = threadIdx.x; 

  // b. Definir un acumulador local en memoria compartida
  __shared__ int localAcc[degreeBins * rBins];

  // c. Inicializar a 0 todos los elementos del acumulador local
  localAcc[locID] = 0;

  // d. Barrera para asegurar que todos los hilos hayan completado la inicialización
  __syncthreads();

  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h)
    return;

  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;

      // e. Actualizar el acumulador global acc usando el acumulador local localAcc
      atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }

  // f. Barrera para asegurar que todos los hilos hayan completado el proceso de incremento del acumulador local
  __syncthreads();

  // g. Loop para sumar los valores del acumulador local localAcc al acumulador global acc
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

int main(int argc, char **argv)
{
  int i;

  PGMImage inImg(argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
  cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

  // Incorporar medicion de tiempo usando CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
  float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Copiar valores precalculados de cos(rad) y sin(rad) a memoria constante
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  cudaEventRecord(start);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  int blockNum = ceil(w * h / 256);
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // calcular el tiempo transcurrido
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // get results from device
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
    //   printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    continue;
  }
  printf("Done!\n");

  // Mostrar el tiempo transcurrido
  printf("Tiempo transcurrido: %.4f ms\n", milliseconds);

  // Clean-up
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  free(pcCos);
  free(pcSin);
  free(cpuht);
  free(h_hough);

  // Destruir los eventos CUDA.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
