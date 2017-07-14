#include "astar.h"

// For debugging
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <iostream> // DEBUG

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t', possible
                                // loss of data
#include <boost/compute.hpp>
#pragma warning(pop)

std::vector<Node> gpuAStar(const Graph &                                     graph,
                           const std::vector<std::pair<Position, Position>> &srcDstList) {
    namespace compute = boost::compute;

    // Set up OpenCL environment and build program
    compute::device gpu = compute::system::default_device();
    std::cout << "OpenCL device: " << gpu.name() << std::endl; // DEBUG

    compute::context       context(gpu);
    compute::command_queue queue(context, gpu);

    auto program = compute::program::create_with_source_file("src/gpuAStar.cl", context);
    program.build();

    // Set up data structures
    static_assert(sizeof(compute::float4_) == 16,
                  "Nodes: represented by four floats (total of 16 bytes)");
    static_assert(sizeof(compute::int2_) == 8,
                  "Adjacency directory: represents the node's set of edges and "
                  "is composed of two non-negative integers (total of 8 bytes)");

    std::vector<compute::float4_> h_nodes;
    std::vector<compute::float4_> h_edges;
    std::vector<compute::uint2_>  h_adjacencyMap;

    auto index = [width = graph.width()](int x, int y) { return y * width + x; };

    for (int y = 0; y < graph.height(); ++y) {
        for (int x = 0; x < graph.width(); ++x) {
            h_nodes.emplace_back((float) index(x, y), (float) x, (float) y, 0.0f);

            const Node current(graph, {x, y});
            const auto begin = h_edges.size();

            for (const auto &neighbor : current.neighbors()) {
                const auto &nbPosition = neighbor.first.position();
                const float nbCost = neighbor.second;

                h_edges.emplace_back((float) index(x, y), (float) index(nbPosition.x, nbPosition.y),
                                     nbCost, 0.0f);
            }

            const auto end = h_edges.size();
            h_adjacencyMap.emplace_back(begin, end);
        }
    }

    // Device memory
    const int maxPathLength =
        2 * (graph.width() + graph.height()) * srcDstList.size(); // TODO: correct size
    compute::vector<compute::float4_> d_nodes(h_nodes.size(), context);
    compute::vector<compute::float4_> d_edges(h_edges.size(), context);
    compute::vector<compute::uint2_>  d_adjacencyMap(h_adjacencyMap.size(), context);
    compute::vector<compute::float4_> d_paths(maxPathLength, context);

    // DEBUG
    auto pb = [](int bytes) {
        if (bytes > (1 << 20))
            return std::to_string(bytes >> 20) + " MBytes";
        if (bytes > (1 << 10))
            return std::to_string(bytes >> 10) + " KBytes";
        return std::to_string(bytes) + " Bytes";
    };

    std::cout << "Memory used:"
              << "\n - Nodes: " << pb(h_nodes.size() * sizeof(compute::float4_))
              << "\n - Edges: " << pb(h_edges.size() * sizeof(compute::float4_))
              << "\n - Adjacency map: " << pb(h_adjacencyMap.size() * sizeof(compute::uint2_))
              << "\n - Paths: " << pb(d_paths.size() * sizeof(compute::float4_)) << std::endl;

    // Create kernel
    compute::kernel kernel(program, "gpuAStar");
    kernel.set_arg(0, d_nodes);
    kernel.set_arg(1, d_nodes.size());
    kernel.set_arg(2, d_edges);
    kernel.set_arg(3, d_edges.size());
    kernel.set_arg(4, d_adjacencyMap);
    kernel.set_arg(5, d_adjacencyMap.size());
    kernel.set_arg(6, d_paths);
    kernel.set_arg(7, d_paths.size());

    // Upload data
    compute::copy(h_nodes.begin(), h_nodes.end(), d_nodes.begin(), queue);
    compute::copy(h_edges.begin(), h_edges.end(), d_edges.begin(), queue);
    compute::copy(h_adjacencyMap.begin(), h_adjacencyMap.end(), d_adjacencyMap.begin(), queue);

    // Run kernel
    const std::size_t globalWorkSize = srcDstList.size();
    const std::size_t localWorkSize = 0;
    queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    queue.finish();

    // Download paths
    std::vector<compute::float4_> h_paths(d_paths.size());
    compute::copy(d_paths.begin(), d_paths.end(), h_paths.begin(), queue);

    return {}; // TODO
}
