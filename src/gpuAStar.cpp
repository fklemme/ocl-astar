#include "astar.h"

#include <algorithm>
#include <boost/compute.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>

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
gpuAStar(const Graph &graph, const std::vector<std::pair<Position, Position>> &srcDstList,
         const boost::compute::device &clDevice) {
    namespace compute = boost::compute;

    const auto numberOfAgents = srcDstList.size();

#ifdef DEBUG_OUTPUT
    const auto maxMemAllocSize = clDevice.get_info<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    const auto maxWorkGroupSize = clDevice.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const auto maxWorkItemDimensions = clDevice.get_info<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    const auto maxWorkItemSizes = clDevice.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    std::cout << "OpenCL device: " << clDevice.name()
              << "\n - Compute units: " << clDevice.compute_units()
              << "\n - Global memory: " << bytes(clDevice.global_memory_size())
              << "\n - Local memory: " << bytes(clDevice.local_memory_size())
              << "\n - Max. memory allocation: " << bytes(maxMemAllocSize)
              << "\n - Max. work group size: " << maxWorkGroupSize << "\n - Max. work item sizes:";
    for (unsigned i = 0; i < maxWorkItemDimensions; ++i)
        std::cout << ' ' << maxWorkItemSizes[i];
    std::cout << std::endl;
#endif

    // Set up OpenCL environment and build program
    compute::context       context(clDevice);
    compute::command_queue queue(context, clDevice);

    auto program = compute::program::create_with_source_file("src/gpuAStar.cl", context);
    program.build(); // Hint: Passing "-O0" somehow prevents compiler crash on AMD

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

    using Info = compute::uint4_; // wrong type, but should be a sufficient placeholder
    static_assert(sizeof(compute::uint_) == sizeof(compute::float_), "Type size check failed!");

    // These should ideally be in local memory, but there is just not enough space!
    compute::vector<uint_float> d_openExt(numberOfAgents * h_nodes.size(), context);
    compute::vector<Info>       d_info(numberOfAgents * h_nodes.size(), context);

    // Not necessarily needed, but comfy
    compute::vector<compute::int2_> d_retCodeLength(numberOfAgents, context);

    // Local memory: Some magic to find a good value for local memory size per agent.
    const auto maxLocalBytes = (std::size_t)(clDevice.local_memory_size() * 0.99); // fails sometimes if you try to allocate 100%
    const auto perAgentTargetBytes = std::max(7 * sizeof(uint_float), (std::size_t)(h_nodes.size() * sizeof(uint_float) * 0.001)); // really hard to pick a good factor here
    const auto perAgentLocalBytes = std::min(perAgentTargetBytes, maxLocalBytes);

    const auto localWorkSize =
        std::min((std::size_t)(1 << (int) std::log2(maxLocalBytes / perAgentLocalBytes)),
                 clDevice.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>() / 2); // FIXME: /2 for notebook.
    const auto globalWorkSize =
        (std::size_t) std::ceil((double) numberOfAgents / localWorkSize) * localWorkSize;

    // We *could* do a reevaluation of perAgentTargetBytes now that we've picked a localWorkSize.
    const auto localMemoryBytes = localWorkSize * perAgentLocalBytes;
    assert(localMemoryBytes <= clDevice.local_memory_size());

    const auto localMemorySize = localMemoryBytes / sizeof(uint_float);
    const auto localMemory = compute::local_buffer<uint_float>(localMemorySize);

#ifdef DEBUG_OUTPUT
    std::cout << "Global memory used:"
              << "\n - Nodes: " << bytes(h_nodes.size() * sizeof(compute::int2_))
              << "\n - Edges: " << bytes(h_edges.size() * sizeof(uint_float))
              << "\n - Adjacency map: " << bytes(h_adjacencyMap.size() * sizeof(compute::uint2_))
              << "\n - SrcDst list: " << bytes(d_srcDstList.size() * sizeof(compute::uint2_))
              << "\n - Paths: " << bytes(d_paths.size() * sizeof(compute::int2_))
              << "\n - Open list (ext): " << bytes(d_openExt.size() * sizeof(uint_float))
              << "\n - Info table: " << bytes(d_info.size() * sizeof(Info))
              << "\nLocal memory used:"
              << "\n - Memory per agent: " << bytes(perAgentLocalBytes)
              << "\n - Local work size: " << localWorkSize
              << "\n - Allocated local memory: " << bytes(localMemory.size() * sizeof(uint_float))
              << std::endl;
#endif

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
    kernel.set_arg<compute::ulong_>(11, localMemorySize / localWorkSize);
    kernel.set_arg(12, d_openExt);
    kernel.set_arg(13, d_info);
    kernel.set_arg(14, d_retCodeLength);

    // This initialization could be done by a kernel as well.
    std::vector<Info> h_info(d_info.size(), {0, 0, 0, 0});

    // Upload data
    const auto uploadStart = std::chrono::high_resolution_clock::now();
    compute::copy(h_nodes.begin(), h_nodes.end(), d_nodes.begin(), queue);
    compute::copy(h_edges.begin(), h_edges.end(), d_edges.begin(), queue);
    compute::copy(h_adjacencyMap.begin(), h_adjacencyMap.end(), d_adjacencyMap.begin(), queue);
    compute::copy(h_srcDstList.begin(), h_srcDstList.end(), d_srcDstList.begin(), queue);
    compute::copy(h_info.begin(), h_info.end(), d_info.begin(), queue);
    const auto uploadStop = std::chrono::high_resolution_clock::now();

    // Run kernel
    const auto kernelStart = std::chrono::high_resolution_clock::now();
    queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    queue.finish();
    const auto kernelStop = std::chrono::high_resolution_clock::now();

    // Download data
    std::vector<compute::int2_> h_paths(d_paths.size()); // x, y
    std::vector<compute::int2_> h_retCodeLength(d_retCodeLength.size());

    const auto downloadStart = std::chrono::high_resolution_clock::now();
    compute::copy(d_paths.begin(), d_paths.end(), h_paths.begin(), queue);
    compute::copy(d_retCodeLength.begin(), d_retCodeLength.end(), h_retCodeLength.begin(), queue);
    const auto downloadStop = std::chrono::high_resolution_clock::now();

    // Convert paths
    std::vector<std::vector<Node>> paths(numberOfAgents);
    for (std::size_t i = 0; i < numberOfAgents; ++i) {
        const int returnCode = h_retCodeLength[i][0];
        const int pathLength = h_retCodeLength[i][1];

        if (returnCode != 0)
            continue;

        paths[i].reserve(pathLength);
        const auto begin = std::next(h_paths.begin(), i * maxPathLength);
        const auto end = std::next(begin, pathLength);

        std::transform(begin, end, std::back_inserter(paths[i]),
                       [&](compute::int2_ node) { return Node(graph, node[0], node[1]); });

        // Path is in inverse order. Reverse it.
        std::reverse(paths[i].begin(), paths[i].end());
    }

    // Print timings
    std::cout << "GPU time for " << numberOfAgents << " runs:"
              << "\n - Upload time: "
              << std::chrono::duration<double>(uploadStop - uploadStart).count() << " seconds"
              << "\n - Kernel runtime: "
              << std::chrono::duration<double>(kernelStop - kernelStart).count() << " seconds"
              << "\n - Download time: "
              << std::chrono::duration<double>(downloadStop - downloadStart).count() << " seconds"
              << std::endl;

    return paths;
}
