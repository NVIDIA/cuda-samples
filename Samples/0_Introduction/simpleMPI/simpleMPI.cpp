/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Simple example demonstrating how to use MPI with CUDA
*
*  Generate some random numbers on one node.
*  Dispatch them to all nodes.
*  Compute their square root on each node's GPU.
*  Compute the average of the results using MPI.
*
*  simpleMPI.cpp: main program, compiled with mpicxx on linux/Mac platforms
*                 on Windows, please download the Microsoft HPC Pack SDK 2008
*/

// MPI include
#include <mpi.h>

// System includes
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

// User include
#include "simpleMPI.h"

// Error handling macros
#define MPI_CHECK(call)                          \
  if ((call) != MPI_SUCCESS) {                   \
    cerr << "MPI error calling \"" #call "\"\n"; \
    my_abort(-1);                                \
  }

// Host code
// No CUDA here, only MPI
int main(int argc, char *argv[]) {
  // Dimensions of the dataset
  int blockSize = 256;
  int gridSize = 10000;
  int dataSizePerNode = gridSize * blockSize;

  // Initialize MPI state
  MPI_CHECK(MPI_Init(&argc, &argv));

  // Get our MPI node number and node count
  int commSize, commRank;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

  // Generate some random numbers on the root node (node 0)
  int dataSizeTotal = dataSizePerNode * commSize;
  float *dataRoot = NULL;

  // Are we the root node?
  if (commRank == 0) {
    cout << "Running on " << commSize << " nodes" << endl;
    dataRoot = new float[dataSizeTotal];
    initData(dataRoot, dataSizeTotal);
  }

  // Allocate a buffer on each node
  float *dataNode = new float[dataSizePerNode];

  // Dispatch a portion of the input data to each node
  MPI_CHECK(MPI_Scatter(dataRoot, dataSizePerNode, MPI_FLOAT, dataNode,
                        dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD));

  if (commRank == 0) {
    // No need for root data any more
    delete[] dataRoot;
  }

  // On each node, run computation on GPU
  computeGPU(dataNode, blockSize, gridSize);

  // Reduction to the root node, computing the sum of output elements
  float sumNode = sum(dataNode, dataSizePerNode);
  float sumRoot;

  MPI_CHECK(
      MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));

  if (commRank == 0) {
    float average = sumRoot / dataSizeTotal;
    cout << "Average of square roots is: " << average << endl;
  }

  // Cleanup
  delete[] dataNode;
  MPI_CHECK(MPI_Finalize());

  if (commRank == 0) {
    cout << "PASSED\n";
  }

  return 0;
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err) {
  cout << "Test FAILED\n";
  MPI_Abort(MPI_COMM_WORLD, err);
}
