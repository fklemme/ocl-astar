#include "astar.h"

// For debugging
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <algorithm>
#include <iostream> // DEBUG
#include <iterator>
#include <limits>
#include <string>

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t', possible
                                // loss of data
#include <boost/compute.hpp>
#pragma warning(pop)

namespace {
// Helper for pritty printing bytes
std::string bytes(unsigned long long bytes) {
    if (bytes > (1 << 20))
        return std::to_string(bytes >> 20) + " MBytes";
    if (bytes > (1 << 10))
        return std::to_string(bytes >> 10) + " KBytes";
    return std::to_string(bytes) + " bytes";
}
} // namespace

std::vector<std::vector<Node>>
gpuAStar(const Graph &graph, const std::vector<std::pair<Position, Position>> &srcDstList) {
    namespace compute = boost::compute;

    const auto numberOfAgents = srcDstList.size();

    // Set up OpenCL environment and build program
    compute::device gpu = compute::system::default_device();

    // DEBUG
    std::cout << "OpenCL device: " << gpu.name() << "\n - Compute units: " << gpu.compute_units()
              << "\n - Global memory: " << bytes(gpu.global_memory_size())
              << "\n - Local memory: " << bytes(gpu.local_memory_size())
              << "\n - Max. memory allocation: "
              << bytes(gpu.get_info<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) << std::endl;

    compute::context       context(gpu);
    compute::command_queue queue(context, gpu);

    auto program = compute::program::create_with_source_file("src/gpuAStar.cl", context);
    program.build();

    // Set up data structures on host
    using uint_float = std::pair<compute::uint_, compute::float_>;
    std::vector<compute::int2_>  h_nodes;        // x, y
    std::vector<uint_float>      h_edges;        // destination index, cost
    std::vector<compute::uint2_> h_adjacencyMap; // edges_begin, edges_end
    std::vector<compute::uint2_> h_srcDstList;   // source index, destination index

    // Convert graph data
    auto index = [width = graph.width()](int x, int y) { return y * width + x; };

    for (int y = 0; y < graph.height(); ++y) {
        for (int x = 0; x < graph.width(); ++x) {
            h_nodes.emplace_back(x, y);

            const Node current(graph, x, y);
            const auto begin = h_edges.size();

            for (const auto &neighbor : current.neighbors()) {
                const auto &nbPosition = neighbor.first.position();
                const float nbCost = neighbor.second;

                h_edges.emplace_back(index(nbPosition.x, nbPosition.y), nbCost);
            }

            const auto end = h_edges.size();
            assert(begin <= std::numeric_limits<compute::uint_>::max());
            assert(end <= std::numeric_limits<compute::uint_>::max());
            h_adjacencyMap.emplace_back((compute::uint_) begin, (compute::uint_) end);
        }
    }

    // Convert source-destination pairs
    for (const auto &srcDst : srcDstList) {
        const auto srcID = index(srcDst.first.x, srcDst.first.y);
        const auto dstID = index(srcDst.second.x, srcDst.second.y);
        h_srcDstList.emplace_back(srcID, dstID);
    }

    // Device memory
    const std::size_t maxPathLength = 2 * (graph.width() + graph.height()); // TODO: correct size
    compute::vector<compute::int2_>  d_nodes(h_nodes.size(), context);
    compute::vector<uint_float>      d_edges(h_edges.size(), context);
    compute::vector<compute::uint2_> d_adjacencyMap(h_adjacencyMap.size(), context);
    compute::vector<compute::uint2_> d_srcDstList(h_srcDstList.size(), context);
    compute::vector<compute::int2_>  d_paths(numberOfAgents * maxPathLength, context);

    // Local memory
    const auto perAgentLocalMemorySize =
        std::min(h_nodes.size() * sizeof(uint_float),
                 (std::size_t) gpu.local_memory_size()); // TODO: correct size
    const auto localWorkSize = perAgentLocalMemorySize / (std::size_t) gpu.local_memory_size();
    assert(localWorkSize >= 1);
    const auto localMemorySize = localWorkSize * perAgentLocalMemorySize;
    assert(localMemorySize <= gpu.local_memory_size());
    const auto localMemory =
        compute::local_buffer<uint_float>(localMemorySize / sizeof(uint_float));

    // Should ideally be in local memory, but probably not enough space!
    compute::vector<uint_float>      d_openExt(numberOfAgents * h_nodes.size(), context);
    compute::vector<compute::float_> d_info(numberOfAgents * h_nodes.size(), context);

    // DEBUG
    std::cout << "Global memory used:"
              << "\n - Nodes: " << bytes(h_nodes.size() * sizeof(compute::int2_))
              << "\n - Edges: " << bytes(h_edges.size() * sizeof(uint_float))
              << "\n - Adjacency map: " << bytes(h_adjacencyMap.size() * sizeof(compute::uint2_))
              << "\n - SrcDst list: " << bytes(d_srcDstList.size() * sizeof(compute::uint2_))
              << "\n - Paths: " << bytes(d_paths.size() * sizeof(compute::int2_))
              << "\n - Open list (ext): " << bytes(d_openExt.size() * sizeof(uint_float))
              << "\n - Info table: " << bytes(d_openExt.size() * sizeof(compute::float_))
              << "\nLocal memory used:"
              << "\n - Memory per agent: " << bytes(perAgentLocalMemorySize)
              << "\n - Local work size: " << localWorkSize
              << "\n - Allocated local memory: " << bytes(localMemorySize) << std::endl;

    // Create kernel
    compute::kernel kernel(program, "gpuAStar");
    kernel.set_arg(0, d_nodes);
    kernel.set_arg<compute::ulong_>(1, d_nodes.size());
    kernel.set_arg(2, d_edges);
    kernel.set_arg<compute::ulong_>(3, d_edges.size());
    kernel.set_arg(4, d_adjacencyMap);
    kernel.set_arg<compute::ulong_>(5, d_adjacencyMap.size());
    kernel.set_arg<compute::ulong_>(6, numberOfAgents);
    kernel.set_arg(7, d_srcDstList);
    kernel.set_arg(8, d_paths);
    kernel.set_arg<compute::ulong_>(9, maxPathLength);
    kernel.set_arg(10, localMemory); // open list
    kernel.set_arg(11, perAgentLocalMemorySize);
    kernel.set_arg(12, d_openExt);
    kernel.set_arg(13, d_info);

    // Upload data
    compute::copy(h_nodes.begin(), h_nodes.end(), d_nodes.begin(), queue);
    compute::copy(h_edges.begin(), h_edges.end(), d_edges.begin(), queue);
    compute::copy(h_adjacencyMap.begin(), h_adjacencyMap.end(), d_adjacencyMap.begin(), queue);
    compute::copy(h_srcDstList.begin(), h_srcDstList.end(), d_srcDstList.begin(), queue);

    // Run kernel
    const std::size_t globalWorkSize = numberOfAgents;
    queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    queue.finish();

    // Download paths
    std::vector<compute::int2_> h_paths(d_paths.size()); // x, y
    compute::copy(d_paths.begin(), d_paths.end(), h_paths.begin(), queue);

    // TODO: Convert paths
    std::vector<std::vector<Node>> paths(numberOfAgents);
    for (std::size_t i = 0; i < numberOfAgents; ++i) {
        const auto begin = std::next(h_paths.begin(), i * maxPathLength);
        const auto end = std::next(begin, maxPathLength);
        const auto source = srcDstList[i].first;
        const auto destination = srcDstList[i].second;

        const Position firstPosition{(int) (*begin)[0], (int) (*begin)[1]};
        if (firstPosition != source)
            continue; // no path found

        for (auto it = begin; it != end; ++it) {
            const Position position{(int) (*it)[0], (int) (*it)[1]};
            paths[i].emplace_back(graph, position);

            // End of path
            if (position == destination)
                break;
        }
    }

    return paths;
}
