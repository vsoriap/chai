/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _CUDA_COMPILER_

#include "support/common.h"
#include <cstdio>

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void SSSP_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end,
    int *threads_run, int *overflow, int *gray_shade,
    int LIMIT, int CPU) {

    extern __shared__ int l_mem[];
    int* base = l_mem;

    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    int *qin, *qout;

    int iter = 1;

    while(*n_t != 0) {

        // Swap queues
        if(iter % 2 == 0) {
            qin  = q1;
            qout = q2;
        } else {
            qin  = q2;
            qout = q1;
        }

        if((*n_t >= LIMIT) | (CPU == 0)) {

            int gray_shade_local = atomicAdd_system(&gray_shade[0], 0);

            // Fetch frontier elements from the queue
            if(tid == 0)
                *base = atomicAdd_system(&head[0], WG_SIZE);
            __syncthreads();

            int my_base = *base;
            while(my_base < *n_t) {
                if(my_base + tid < *n_t && *overflow == 0) {
                    // Visit a node from the current frontier
                    int pid = qin[my_base + tid];
                    //////////////// Visit node ///////////////////////////
                    atomicExch_system(&color[pid], BLACK); // Node visited
                    int  cur_cost = atomicAdd_system(&cost[pid], 0); // Look up shortest-path distance to this node
                    Node cur_node;
                    cur_node.x = graph_nodes_av[pid].x;
                    cur_node.y = graph_nodes_av[pid].y;
                    Edge cur_edge;
                    // For each outgoing edge
                    for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                        cur_edge.x = graph_edges_av[i].x;
                        cur_edge.y = graph_edges_av[i].y;
                        int id     = cur_edge.x;
                        int cost_local   = cur_edge.y;
                        cost_local += cur_cost;
                        int orig_cost = atomicMax_system(&cost[id], cost_local);
                        if(orig_cost < cost_local) {
                            int old_color = atomicMax_system(&color[id], gray_shade_local);
                            if(old_color != gray_shade_local) {
                                // Push to the queue
                                int index_o   = atomicAdd_system(&tail[0],1);
                                qout[index_o] = id;
                            }
                        }
                    }
                }
                if(tid == 0)
                    *base = atomicAdd_system(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
                __syncthreads();
                my_base = *base;
            }
        }

        if(CPU) { // if CPU is available
            iter++;
            if(tid == 0) {
                atomicAdd_system(&threads_end[0], WG_SIZE);

                __threadfence();

                while(atomicAdd_system(&threads_run[0], 0) < iter) {
                }
            }
        } else { // if GPU only
            iter++;
            if(tid == 0)
                atomicAdd_system(&threads_end[0], WG_SIZE);
            __threadfence();
            if(gtid == 0) {
                while(atomicAdd_system(&threads_end[0], 0) != MAXWG * WG_SIZE) {
                }
                *n_t = atomicAdd_system(&tail[0], 0);
                atomicExch_system(&tail[0], 0);
                atomicExch_system(&head[0], 0);
                atomicExch_system(&threads_end[0], 0);
                if(iter % 2 == 0)
                    atomicExch_system(&gray_shade[0], GRAY0);
                else
                    atomicExch_system(&gray_shade[0], GRAY1);
                __threadfence();
                atomicAdd_system(&threads_run[0], 1);
                //printf("Iter %d, n_t %d\n",iter, *n_t);
            }
            if(tid == 0 && gtid != 0) {
                while(atomicAdd_system(&threads_run[0], 0) < iter) {
                }
            }
        }
        __syncthreads();
    }
}

cudaError_t call_SSSP_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int LIMIT, const int CPU, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    SSSP_gpu<<<dimGrid, dimBlock, l_mem_size>>>(graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, gray_shade, LIMIT, CPU);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
