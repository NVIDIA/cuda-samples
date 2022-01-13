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

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// A structure of 2D points (structure of arrays).
////////////////////////////////////////////////////////////////////////////////
class Points {
  float *m_x;
  float *m_y;

 public:
  // Constructor.
  __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}

  // Constructor.
  __host__ __device__ Points(float *x, float *y) : m_x(x), m_y(y) {}

  // Get a point.
  __host__ __device__ __forceinline__ float2 get_point(int idx) const {
    return make_float2(m_x[idx], m_y[idx]);
  }

  // Set a point.
  __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p) {
    m_x[idx] = p.x;
    m_y[idx] = p.y;
  }

  // Set the pointers.
  __host__ __device__ __forceinline__ void set(float *x, float *y) {
    m_x = x;
    m_y = y;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A 2D bounding box
////////////////////////////////////////////////////////////////////////////////
class Bounding_box {
  // Extreme points of the bounding box.
  float2 m_p_min;
  float2 m_p_max;

 public:
  // Constructor. Create a unit box.
  __host__ __device__ Bounding_box() {
    m_p_min = make_float2(0.0f, 0.0f);
    m_p_max = make_float2(1.0f, 1.0f);
  }

  // Compute the center of the bounding-box.
  __host__ __device__ void compute_center(float2 &center) const {
    center.x = 0.5f * (m_p_min.x + m_p_max.x);
    center.y = 0.5f * (m_p_min.y + m_p_max.y);
  }

  // The points of the box.
  __host__ __device__ __forceinline__ const float2 &get_max() const {
    return m_p_max;
  }

  __host__ __device__ __forceinline__ const float2 &get_min() const {
    return m_p_min;
  }

  // Does a box contain a point.
  __host__ __device__ bool contains(const float2 &p) const {
    return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y &&
           p.y < m_p_max.y;
  }

  // Define the bounding box.
  __host__ __device__ void set(float min_x, float min_y, float max_x,
                               float max_y) {
    m_p_min.x = min_x;
    m_p_min.y = min_y;
    m_p_max.x = max_x;
    m_p_max.y = max_y;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A node of a quadree.
////////////////////////////////////////////////////////////////////////////////
class Quadtree_node {
  // The identifier of the node.
  int m_id;
  // The bounding box of the tree.
  Bounding_box m_bounding_box;
  // The range of points.
  int m_begin, m_end;

 public:
  // Constructor.
  __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}

  // The ID of a node at its level.
  __host__ __device__ int id() const { return m_id; }

  // The ID of a node at its level.
  __host__ __device__ void set_id(int new_id) { m_id = new_id; }

  // The bounding box.
  __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const {
    return m_bounding_box;
  }

  // Set the bounding box.
  __host__ __device__ __forceinline__ void set_bounding_box(float min_x,
                                                            float min_y,
                                                            float max_x,
                                                            float max_y) {
    m_bounding_box.set(min_x, min_y, max_x, max_y);
  }

  // The number of points in the tree.
  __host__ __device__ __forceinline__ int num_points() const {
    return m_end - m_begin;
  }

  // The range of points in the tree.
  __host__ __device__ __forceinline__ int points_begin() const {
    return m_begin;
  }

  __host__ __device__ __forceinline__ int points_end() const { return m_end; }

  // Define the range for that node.
  __host__ __device__ __forceinline__ void set_range(int begin, int end) {
    m_begin = begin;
    m_end = end;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Algorithm parameters.
////////////////////////////////////////////////////////////////////////////////
struct Parameters {
  // Choose the right set of points to use as in/out.
  int point_selector;
  // The number of nodes at a given level (2^k for level k).
  int num_nodes_at_this_level;
  // The recursion depth.
  int depth;
  // The max value for depth.
  const int max_depth;
  // The minimum number of points in a node to stop recursion.
  const int min_points_per_node;

  // Constructor set to default values.
  __host__ __device__ Parameters(int max_depth, int min_points_per_node)
      : point_selector(0),
        num_nodes_at_this_level(1),
        depth(0),
        max_depth(max_depth),
        min_points_per_node(min_points_per_node) {}

  // Copy constructor. Changes the values for next iteration.
  __host__ __device__ Parameters(const Parameters &params, bool)
      : point_selector((params.point_selector + 1) % 2),
        num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
        depth(params.depth + 1),
        max_depth(params.max_depth),
        min_points_per_node(params.min_points_per_node) {}
};

////////////////////////////////////////////////////////////////////////////////
// Build a quadtree on the GPU. Use CUDA Dynamic Parallelism.
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded or the minimum number of points is
// reached. The threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the quadtree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leavin (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into four geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each quadrant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in that sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 4 children, it
// remains to dispatch the points. It is straightforward.
//
// 5- Launch new blocks.
//
// The block launches four new blocks: One per children. Each of the four blocks
// will apply the same algorithm.
////////////////////////////////////////////////////////////////////////////////
template <int NUM_THREADS_PER_BLOCK>
__global__ void build_quadtree_kernel(Quadtree_node *nodes, Points *points,
                                      Parameters params) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // The number of warps in a block.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

  // Shared memory to store the number of points.
  extern __shared__ int smem[];

  // s_num_pts[4][NUM_WARPS_PER_BLOCK];
  // Addresses of shared memory.
  volatile int *s_num_pts[4];

  for (int i = 0; i < 4; ++i)
    s_num_pts[i] = (volatile int *)&smem[i * NUM_WARPS_PER_BLOCK];

  // Compute the coordinates of the threads in the block.
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  // Mask for compaction.
  // Same as: asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );
  int lane_mask_lt = (1 << lane_id) - 1;

  // The current node.
  Quadtree_node &node = nodes[blockIdx.x];

  // The number of points in the node.
  int num_points = node.num_points();

  float2 center;
  int range_begin, range_end;
  int warp_cnts[4] = {0, 0, 0, 0};
  //
  // 1- Check the number of points and its depth.
  //

  // Stop the recursion here. Make sure points[0] contains all the points.
  if (params.depth >= params.max_depth ||
      num_points <= params.min_points_per_node) {
    if (params.point_selector == 1) {
      int it = node.points_begin(), end = node.points_end();

      for (it += threadIdx.x; it < end; it += NUM_THREADS_PER_BLOCK)
        if (it < end) points[0].set_point(it, points[1].get_point(it));
    }

    return;
  }

  // Compute the center of the bounding box of the points.
  const Bounding_box &bbox = node.bounding_box();

  bbox.compute_center(center);

  // Find how many points to give to each warp.
  int num_points_per_warp = max(
      warpSize, (num_points + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);

  // Each warp of threads will compute the number of points to move to each
  // quadrant.
  range_begin = node.points_begin() + warp_id * num_points_per_warp;
  range_end = min(range_begin + num_points_per_warp, node.points_end());

  //
  // 2- Count the number of points in each child.
  //

  // Input points.
  const Points &in_points = points[params.point_selector];

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  // Compute the number of points.
  for (int range_it = range_begin + tile32.thread_rank();
       tile32.any(range_it < range_end); range_it += warpSize) {
    // Is it still an active thread?
    bool is_active = range_it < range_end;

    // Load the coordinates of the point.
    float2 p =
        is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

    // Count top-left points.
    int num_pts =
        __popc(tile32.ballot(is_active && p.x < center.x && p.y >= center.y));
    warp_cnts[0] += tile32.shfl(num_pts, 0);

    // Count top-right points.
    num_pts =
        __popc(tile32.ballot(is_active && p.x >= center.x && p.y >= center.y));
    warp_cnts[1] += tile32.shfl(num_pts, 0);

    // Count bottom-left points.
    num_pts =
        __popc(tile32.ballot(is_active && p.x < center.x && p.y < center.y));
    warp_cnts[2] += tile32.shfl(num_pts, 0);

    // Count bottom-right points.
    num_pts =
        __popc(tile32.ballot(is_active && p.x >= center.x && p.y < center.y));
    warp_cnts[3] += tile32.shfl(num_pts, 0);
  }

  if (tile32.thread_rank() == 0) {
    s_num_pts[0][warp_id] = warp_cnts[0];
    s_num_pts[1][warp_id] = warp_cnts[1];
    s_num_pts[2][warp_id] = warp_cnts[2];
    s_num_pts[3][warp_id] = warp_cnts[3];
  }

  // Make sure warps have finished counting.
  cg::sync(cta);

  //
  // 3- Scan the warps' results to know the "global" numbers.
  //

  // First 4 warps scan the numbers of points per child (inclusive scan).
  if (warp_id < 4) {
    int num_pts = tile32.thread_rank() < NUM_WARPS_PER_BLOCK
                      ? s_num_pts[warp_id][tile32.thread_rank()]
                      : 0;
#pragma unroll

    for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset *= 2) {
      int n = tile32.shfl_up(num_pts, offset);

      if (tile32.thread_rank() >= offset) num_pts += n;
    }

    if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
      s_num_pts[warp_id][tile32.thread_rank()] = num_pts;
  }

  cg::sync(cta);

  // Compute global offsets.
  if (warp_id == 0) {
    int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];

    for (int row = 1; row < 4; ++row) {
      int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK - 1];
      cg::sync(tile32);

      if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
        s_num_pts[row][tile32.thread_rank()] += sum;

      cg::sync(tile32);
      sum += tmp;
    }
  }

  cg::sync(cta);

  // Make the scan exclusive.
  int val = 0;
  if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK) {
    val = threadIdx.x == 0 ? 0 : smem[threadIdx.x - 1];
    val += node.points_begin();
  }

  cg::sync(cta);

  if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK) {
    smem[threadIdx.x] = val;
  }

  cg::sync(cta);

  //
  // 4- Move points.
  //
  if (!(params.depth >= params.max_depth ||
        num_points <= params.min_points_per_node)) {
    // Output points.
    Points &out_points = points[(params.point_selector + 1) % 2];

    warp_cnts[0] = s_num_pts[0][warp_id];
    warp_cnts[1] = s_num_pts[1][warp_id];
    warp_cnts[2] = s_num_pts[2][warp_id];
    warp_cnts[3] = s_num_pts[3][warp_id];

    const Points &in_points = points[params.point_selector];
    // Reorder points.
    for (int range_it = range_begin + tile32.thread_rank();
         tile32.any(range_it < range_end); range_it += warpSize) {
      // Is it still an active thread?
      bool is_active = range_it < range_end;

      // Load the coordinates of the point.
      float2 p =
          is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

      // Count top-left points.
      bool pred = is_active && p.x < center.x && p.y >= center.y;
      int vote = tile32.ballot(pred);
      int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);

      if (pred) out_points.set_point(dest, p);

      warp_cnts[0] += tile32.shfl(__popc(vote), 0);

      // Count top-right points.
      pred = is_active && p.x >= center.x && p.y >= center.y;
      vote = tile32.ballot(pred);
      dest = warp_cnts[1] + __popc(vote & lane_mask_lt);

      if (pred) out_points.set_point(dest, p);

      warp_cnts[1] += tile32.shfl(__popc(vote), 0);

      // Count bottom-left points.
      pred = is_active && p.x < center.x && p.y < center.y;
      vote = tile32.ballot(pred);
      dest = warp_cnts[2] + __popc(vote & lane_mask_lt);

      if (pred) out_points.set_point(dest, p);

      warp_cnts[2] += tile32.shfl(__popc(vote), 0);

      // Count bottom-right points.
      pred = is_active && p.x >= center.x && p.y < center.y;
      vote = tile32.ballot(pred);
      dest = warp_cnts[3] + __popc(vote & lane_mask_lt);

      if (pred) out_points.set_point(dest, p);

      warp_cnts[3] += tile32.shfl(__popc(vote), 0);
    }
  }

  cg::sync(cta);

  if (tile32.thread_rank() == 0) {
    s_num_pts[0][warp_id] = warp_cnts[0];
    s_num_pts[1][warp_id] = warp_cnts[1];
    s_num_pts[2][warp_id] = warp_cnts[2];
    s_num_pts[3][warp_id] = warp_cnts[3];
  }

  cg::sync(cta);

  //
  // 5- Launch new blocks.
  //
  if (!(params.depth >= params.max_depth ||
        num_points <= params.min_points_per_node)) {
    // The last thread launches new blocks.
    if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1) {
      // The children.
      Quadtree_node *children =
          &nodes[params.num_nodes_at_this_level - (node.id() & ~3)];

      // The offsets of the children at their level.
      int child_offset = 4 * node.id();

      // Set IDs.
      children[child_offset + 0].set_id(4 * node.id() + 0);
      children[child_offset + 1].set_id(4 * node.id() + 1);
      children[child_offset + 2].set_id(4 * node.id() + 2);
      children[child_offset + 3].set_id(4 * node.id() + 3);

      const Bounding_box &bbox = node.bounding_box();
      // Points of the bounding-box.
      const float2 &p_min = bbox.get_min();
      const float2 &p_max = bbox.get_max();

      // Set the bounding boxes of the children.
      children[child_offset + 0].set_bounding_box(p_min.x, center.y, center.x,
                                                  p_max.y);  // Top-left.
      children[child_offset + 1].set_bounding_box(center.x, center.y, p_max.x,
                                                  p_max.y);  // Top-right.
      children[child_offset + 2].set_bounding_box(p_min.x, p_min.y, center.x,
                                                  center.y);  // Bottom-left.
      children[child_offset + 3].set_bounding_box(center.x, p_min.y, p_max.x,
                                                  center.y);  // Bottom-right.

      // Set the ranges of the children.

      children[child_offset + 0].set_range(node.points_begin(),
                                           s_num_pts[0][warp_id]);
      children[child_offset + 1].set_range(s_num_pts[0][warp_id],
                                           s_num_pts[1][warp_id]);
      children[child_offset + 2].set_range(s_num_pts[1][warp_id],
                                           s_num_pts[2][warp_id]);
      children[child_offset + 3].set_range(s_num_pts[2][warp_id],
                                           s_num_pts[3][warp_id]);

      // Launch 4 children.
      build_quadtree_kernel<NUM_THREADS_PER_BLOCK><<<
          4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
          &children[child_offset], points, Parameters(params, true));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Make sure a Quadtree is properly defined.
////////////////////////////////////////////////////////////////////////////////
bool check_quadtree(const Quadtree_node *nodes, int idx, int num_pts,
                    Points *pts, Parameters params) {
  const Quadtree_node &node = nodes[idx];
  int num_points = node.num_points();

  if (!(params.depth == params.max_depth ||
        num_points <= params.min_points_per_node)) {
    int num_points_in_children = 0;

    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 4 * idx + 0].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 4 * idx + 1].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 4 * idx + 2].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 4 * idx + 3].num_points();

    if (num_points_in_children != node.num_points()) return false;

    return check_quadtree(&nodes[params.num_nodes_at_this_level], 4 * idx + 0,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 4 * idx + 1,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 4 * idx + 2,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 4 * idx + 3,
                          num_pts, pts, Parameters(params, true));
  }

  const Bounding_box &bbox = node.bounding_box();

  for (int it = node.points_begin(); it < node.points_end(); ++it) {
    if (it >= num_pts) return false;

    float2 p = pts->get_point(it);

    if (!bbox.contains(p)) return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Parallel random number generator.
////////////////////////////////////////////////////////////////////////////////
struct Random_generator {
  int count;

  __host__ __device__ Random_generator() : count(0) {}
  __host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()() {
#ifdef __CUDA_ARCH__
    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
    // thrust::generate may call operator() more than once per thread.
    // Hence, increment count by grid size to ensure uniqueness of seed
    count += blockDim.x * gridDim.x;
#else
    unsigned seed = hash(0);
#endif
    thrust::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;
    return thrust::make_tuple(distrib(rng), distrib(rng));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Allocate GPU structs, launch kernel and clean up
////////////////////////////////////////////////////////////////////////////////
bool cdpQuadtree(int warp_size) {
  // Constants to control the algorithm.
  const int num_points = 1024;
  const int max_depth = 8;
  const int min_points_per_node = 16;

  // Allocate memory for points.
  thrust::device_vector<float> x_d0(num_points);
  thrust::device_vector<float> x_d1(num_points);
  thrust::device_vector<float> y_d0(num_points);
  thrust::device_vector<float> y_d1(num_points);

  // Generate random points.
  Random_generator rnd;
  thrust::generate(
      thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(), y_d0.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(x_d0.end(), y_d0.end())),
      rnd);

  // Host structures to analyze the device ones.
  Points points_init[2];
  points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]),
                     thrust::raw_pointer_cast(&y_d0[0]));
  points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]),
                     thrust::raw_pointer_cast(&y_d1[0]));

  // Allocate memory to store points.
  Points *points;
  checkCudaErrors(cudaMalloc((void **)&points, 2 * sizeof(Points)));
  checkCudaErrors(cudaMemcpy(points, points_init, 2 * sizeof(Points),
                             cudaMemcpyHostToDevice));

  // We could use a close form...
  int max_nodes = 0;

  for (int i = 0, num_nodes_at_level = 1; i < max_depth;
       ++i, num_nodes_at_level *= 4)
    max_nodes += num_nodes_at_level;

  // Allocate memory to store the tree.
  Quadtree_node root;
  root.set_range(0, num_points);
  Quadtree_node *nodes;
  checkCudaErrors(
      cudaMalloc((void **)&nodes, max_nodes * sizeof(Quadtree_node)));
  checkCudaErrors(
      cudaMemcpy(nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice));

  // We set the recursion limit for CDP to max_depth.
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

  // Build the quadtree.
  Parameters params(max_depth, min_points_per_node);
  std::cout << "Launching CDP kernel to build the quadtree" << std::endl;
  const int NUM_THREADS_PER_BLOCK = 128;  // Do not use less than 128 threads.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
  const size_t smem_size = 4 * NUM_WARPS_PER_BLOCK * sizeof(int);
  build_quadtree_kernel<
      NUM_THREADS_PER_BLOCK><<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(
      nodes, points, params);
  checkCudaErrors(cudaGetLastError());

  // Copy points to CPU.
  thrust::host_vector<float> x_h(x_d0);
  thrust::host_vector<float> y_h(y_d0);
  Points host_points;
  host_points.set(thrust::raw_pointer_cast(&x_h[0]),
                  thrust::raw_pointer_cast(&y_h[0]));

  // Copy nodes to CPU.
  Quadtree_node *host_nodes = new Quadtree_node[max_nodes];
  checkCudaErrors(cudaMemcpy(host_nodes, nodes,
                             max_nodes * sizeof(Quadtree_node),
                             cudaMemcpyDeviceToHost));

  // Validate the results.
  bool ok = check_quadtree(host_nodes, 0, num_points, &host_points, params);
  std::cout << "Results: " << (ok ? "OK" : "FAILED") << std::endl;

  // Free CPU memory.
  delete[] host_nodes;

  // Free memory.
  checkCudaErrors(cudaFree(nodes));
  checkCudaErrors(cudaFree(points));

  return ok;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Find/set the device.
  // The test requires an architecture SM35 or greater (CDP capable).
  int cuda_device = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
  int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) ||
                   deviceProps.major >= 4;

  printf("GPU device %s has compute capabilities (SM %d.%d)\n",
         deviceProps.name, deviceProps.major, deviceProps.minor);

  if (!cdpCapable) {
    std::cerr << "cdpQuadTree requires SM 3.5 or higher to use CUDA Dynamic "
                 "Parallelism.  Exiting...\n"
              << std::endl;
    exit(EXIT_WAIVED);
  }

  bool ok = cdpQuadtree(deviceProps.warpSize);

  return (ok ? EXIT_SUCCESS : EXIT_FAILURE);
}
