# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
PageRank Algorithm

Demonstrates GPU-accelerated PageRank computation for graph analysis:
- Graph representation using edge lists and cuDF DataFrames
- GPU-optimized PageRank via RAPIDS cuGraph library
- Performance comparison: cuGraph GPU vs CPU reference

Uses RAPIDS cuGraph for production-grade graph analytics on GPU.

PageRank Algorithm:
    PR(v) = (1-d)/N + d * sum(PR(u)/out_degree(u)) for all u linking to v
    where d = damping factor (typically 0.85), N = number of nodes
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import print_gpu_info, verify_array_result  # noqa: E402

try:
    import cudf
    import cugraph
    import cupy as cp
    import numpy as np
    from cuda.core import Device, EventOptions, Stream
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


def generate_random_graph(
    num_nodes: int,
    avg_edges_per_node: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random directed graph as edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph
    avg_edges_per_node : int
        Average number of outgoing edges per node
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (sources, destinations, out_degree) arrays
    """
    rng = np.random.default_rng(seed)

    sources_list: list[int] = []
    destinations_list: list[int] = []
    out_degree = np.zeros(num_nodes, dtype=np.int32)

    for src in range(num_nodes):
        # Random number of outgoing edges (Poisson distribution)
        n_edges = max(1, rng.poisson(avg_edges_per_node))
        n_edges = min(n_edges, num_nodes - 1)
        # Random destinations (no self-loops); rejection sampling avoids O(N²) memory
        dests: set[int] = set()
        while len(dests) < n_edges:
            d = int(rng.integers(0, num_nodes))
            if d != src:
                dests.add(d)
        dests = np.array(list(dests), dtype=np.int32)
        for dst in dests:
            sources_list.append(src)
            destinations_list.append(dst)
        out_degree[src] = len(dests)

    sources = np.array(sources_list, dtype=np.int32)
    destinations = np.array(destinations_list, dtype=np.int32)

    return sources, destinations, out_degree


def pagerank_cpu(
    sources: np.ndarray,
    destinations: np.ndarray,
    out_degree: np.ndarray,
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """
    Compute PageRank on CPU using iterative method.

    Parameters
    ----------
    sources : np.ndarray
        Source nodes of edges
    destinations : np.ndarray
        Destination nodes of edges
    out_degree : np.ndarray
        Outgoing degree for each node
    num_nodes : int
        Number of nodes
    damping : float
        Damping factor (default: 0.85)
    max_iterations : int
        Maximum iterations
    tolerance : float
        Convergence tolerance

    Returns
    -------
    tuple[np.ndarray, int]
        (PageRank scores, iterations until convergence)
    """
    # Build incoming edges list for each node
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in zip(sources, destinations):
        incoming[dst].append(src)

    # Initialize PageRank uniformly
    pr = np.ones(num_nodes, dtype=np.float32) / num_nodes
    pr_new = np.zeros(num_nodes, dtype=np.float32)

    base_score = (1.0 - damping) / num_nodes

    for iteration in range(max_iterations):
        # Handle dangling nodes (nodes with no outgoing edges)
        dangling_sum = np.sum(pr[out_degree == 0])
        dangling_contrib = damping * dangling_sum / num_nodes

        for v in range(num_nodes):
            # Sum contributions from incoming neighbors
            incoming_sum = 0.0
            for u in incoming[v]:
                if out_degree[u] > 0:
                    incoming_sum += pr[u] / out_degree[u]

            pr_new[v] = base_score + damping * incoming_sum + dangling_contrib

        # Check convergence
        diff = np.sum(np.abs(pr_new - pr))
        pr, pr_new = pr_new, pr

        if diff < tolerance:
            return pr, iteration + 1

    return pr, max_iterations


def run_pagerank_benchmark(
    num_nodes: int = 10000,
    avg_edges: int = 15,
    max_iterations: int = 100,
) -> bool:
    """
    Run PageRank benchmark comparing cuGraph GPU and CPU performance.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph
    avg_edges : int
        Average edges per node
    max_iterations : int
        Maximum PageRank iterations

    Returns
    -------
    bool
        True if benchmark succeeded
    """
    print("=" * 60)
    print("PageRank Algorithm (using RAPIDS cuGraph)")
    print("=" * 60)

    # Initialize cuda.core device and stream
    device = Device(0)
    device.set_current()
    stream: Stream = device.create_stream()
    print()
    print_gpu_info(device)

    # Make CuPy/cuDF use our cuda.core stream
    cp.cuda.ExternalStream(int(stream.handle)).use()

    # Generate random graph
    print("\nGraph Parameters:")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Avg edges/node: {avg_edges}")

    sources, destinations, out_degree = generate_random_graph(
        num_nodes, avg_edges, seed=42
    )

    total_edges = len(sources)
    print(f"  Total edges: {total_edges:,}")
    print(f"  Avg in-degree: {total_edges / num_nodes:.1f}")

    # -------------------------------------------------------------------------
    # GPU PageRank (cuGraph)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("GPU PageRank (RAPIDS cuGraph)")
    print("-" * 60)

    # Create cuGraph graph from edge list with store_transposed for optimal perf
    gdf = cudf.DataFrame(
        {
            "src": sources,
            "dst": destinations,
        }
    )
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="src", destination="dst", store_transposed=True)

    event_opts = EventOptions(enable_timing=True)

    try:
        # Warmup
        _ = cugraph.pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)
        stream.sync()

        # Timed run using cuda.core events
        start = stream.record(options=event_opts)
        pr_result = cugraph.pagerank(G, alpha=0.85, max_iter=max_iterations, tol=1e-6)
        end = stream.record(options=event_opts)
        end.sync()

        gpu_time_ms = end - start
        print(f"Time: {gpu_time_ms:.3f} ms")

        # Extract results sorted by vertex ID (to numpy for verification)
        pr_df = pr_result.sort_values("vertex").reset_index(drop=True)
        pr_gpu = pr_df["pagerank"].to_numpy()

        # Show top PageRank nodes
        top_k = 5
        top_df = pr_result.nlargest(top_k, "pagerank")
        print(f"\nTop {top_k} nodes by PageRank:")
        for i, row in enumerate(top_df.to_pandas().itertuples()):
            print(f"  {i + 1}. Node {row.vertex:5d}: {row.pagerank:.6f}")

        # -------------------------------------------------------------------------
        # CPU PageRank
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("CPU PageRank (Reference)")
        print("-" * 60)

        cpu_start = time.perf_counter()
        pr_cpu, cpu_iters = pagerank_cpu(
            sources, destinations, out_degree, num_nodes, max_iterations=max_iterations
        )
        cpu_end = time.perf_counter()

        cpu_time_ms = (cpu_end - cpu_start) * 1000
        print(f"Time: {cpu_time_ms:.3f} ms")
        print(f"Iterations: {cpu_iters}")

        # -------------------------------------------------------------------------
        # Performance Summary
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("PERFORMANCE SUMMARY")
        print("-" * 60)
        speedup = cpu_time_ms / gpu_time_ms
        print(f"GPU (cuGraph): {gpu_time_ms:.3f} ms")
        print(f"CPU (Reference): {cpu_time_ms:.3f} ms")
        print(f"Speedup: {speedup:.1f}x")

        # -------------------------------------------------------------------------
        # Verification
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("VERIFICATION")
        print("-" * 60)

        # Compare GPU and CPU results (cuGraph and CPU ref may converge differently)
        print("GPU vs CPU PageRank scores: ", end="")
        success = verify_array_result(
            pr_gpu, pr_cpu, rtol=1e-2, atol=1e-4, verbose=True
        )

        # Verify PageRank properties
        print("\nPageRank Properties:")
        pr_sum = float(np.sum(pr_gpu))
        print(f"  Sum of scores: {pr_sum:.6f} (should be ~1.0)")

        pr_min = float(np.min(pr_gpu))
        pr_max = float(np.max(pr_gpu))
        print(f"  Min score: {pr_min:.6f}")
        print(f"  Max score: {pr_max:.6f}")

        # Check that sum is approximately 1
        sum_ok = abs(pr_sum - 1.0) < 0.01
        print(f"  Sum check: {'✓' if sum_ok else '✗'}")

        success = success and sum_ok
        return success
    finally:
        cp.cuda.Stream.null.use()
        stream.close()


def main() -> None:
    """Entry point."""
    success = run_pagerank_benchmark()
    if success:
        print("\nDone")
    else:
        print("\nBenchmark completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
