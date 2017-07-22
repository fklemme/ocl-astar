#include "astar.h"

#include <chrono>
#include <iostream>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t',
                                // possible loss of data
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

std::vector<Node> gpuGAStar(const Graph &graph, const Position &source,
                            const Position &destination) {
    namespace compute = boost::compute;

    // Just so we don't have to handle this case in the kernels...
    if (source == destination)
        return {{graph, destination}};

    const std::size_t numberOfQueues = 64; // TODO: How to pick this number?
    const std::size_t sizeOfAQueue = graph.size() / numberOfQueues + 1;
    assert(sizeOfAQueue <= std::numeric_limits<compute::uint_>::max());

    // Select default OpenCL device
    compute::device clDevice = compute::system::default_device();

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

    auto program = compute::program::create_with_source_file("src/gpuGAStar.cl", context);
    program.build();

    // Set up data structures on host
    // Let's just use similar strucutures to the other GPU A* implementation.
    using uint_float = std::pair<compute::uint_, compute::float_>;
    std::vector<compute::int2_>  h_nodes;        // x, y
    std::vector<uint_float>      h_edges;        // destination index, cost
    std::vector<compute::uint2_> h_adjacencyMap; // edges_begin, edges_end

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

    // Device memory
    const std::size_t maxPathLength = 2 * (graph.width() + graph.height()); // TODO: correct size
    compute::vector<compute::int2_>  d_nodes(h_nodes.size(), context);
    compute::vector<uint_float>      d_edges(h_edges.size(), context);
    compute::vector<compute::uint2_> d_adjacencyMap(h_adjacencyMap.size(), context);
    compute::vector<compute::int2_>  d_path(maxPathLength, context);
    compute::vector<uint_float>      d_openLists(numberOfQueues * sizeOfAQueue, context);
    compute::vector<compute::uint_>  d_openSizes(numberOfQueues, context);
    compute::vector<compute::char_>  d_closed(h_nodes.size(), context);

#ifdef DEBUG_OUTPUT
    std::cout << "Global memory used:"
              << "\n - Nodes: " << bytes(h_nodes.size() * sizeof(compute::int2_))
              << "\n - Edges: " << bytes(h_edges.size() * sizeof(uint_float))
              << "\n - Adjacency map: " << bytes(h_adjacencyMap.size() * sizeof(compute::uint2_))
              << "\n - Path: " << bytes(d_path.size() * sizeof(compute::int2_))
              << "\n - Open lists: " << bytes(d_openLists.size() * sizeof(uint_float))
              << "\n - Open list sizes: " << bytes(d_openSizes.size() * sizeof(compute::uint_))
              << "\n - Closed list: " << bytes(d_closed.size() * sizeof(compute::char_))
              << std::endl;
#endif
    // Create kernels
    compute::kernel extractAndExpand(program, "extractAndExpand");
    compute::kernel checkAndFinalize(program, "checkAndFinalize");
    compute::kernel duplicateDetection(program, "duplicateDetection");
    compute::kernel computeAndPushBack(program, "computeAndPushBack");

    // Set kernel arguments
    extractAndExpand.set_arg<compute::ulong_>(0, numberOfQueues);
    extractAndExpand.set_arg<compute::ulong_>(1, sizeOfAQueue);
    extractAndExpand.set_arg(2, d_openLists);
    extractAndExpand.set_arg(3, d_openSizes);

    // Data initialization
    std::vector<uint_float>     h_openLists(1, std::make_pair(index(source.x, source.y), 0.0f));
    std::vector<compute::uint_> h_openSizes(d_openSizes.size(), 0);
    h_openSizes.front() = 1; // the first list contains one node: source
    std::vector<compute::char_> h_closed(d_closed.size(), 0);

    // Upload data
    const auto uploadStart = std::chrono::high_resolution_clock::now();
    compute::copy(h_nodes.begin(), h_nodes.end(), d_nodes.begin(), queue);
    compute::copy(h_edges.begin(), h_edges.end(), d_edges.begin(), queue);
    compute::copy(h_adjacencyMap.begin(), h_adjacencyMap.end(), d_adjacencyMap.begin(), queue);
    compute::copy(h_openLists.begin(), h_openLists.end(), d_openLists.begin(), queue); // source
    compute::copy(h_openSizes.begin(), h_openSizes.end(), d_openSizes.begin(), queue);
    compute::copy(h_closed.begin(), h_closed.end(), d_closed.begin(), queue);
    const auto uploadStop = std::chrono::high_resolution_clock::now();

    // TODO: Figure these out!
    const std::size_t globalWorkSize = numberOfQueues;
    const std::size_t localWorkSize = numberOfQueues;

    // Run kernels
    const auto kernelsStart = std::chrono::high_resolution_clock::now();
    while (true) {
        queue.enqueue_1d_range_kernel(extractAndExpand, 0, globalWorkSize, localWorkSize);
        queue.enqueue_1d_range_kernel(checkAndFinalize, 0, globalWorkSize, localWorkSize);
        queue.finish();
        // TODO: download info and break
        break;
        queue.enqueue_1d_range_kernel(duplicateDetection, 0, globalWorkSize, localWorkSize);
        queue.enqueue_1d_range_kernel(computeAndPushBack, 0, globalWorkSize, localWorkSize);
    }
    const auto kernelsStop = std::chrono::high_resolution_clock::now();

    // Download data
    std::vector<compute::int2_> h_path(d_path.size());

    const auto downloadStart = std::chrono::high_resolution_clock::now();
    compute::copy(d_path.begin(), d_path.end(), h_path.begin(), queue);
    const auto downloadStop = std::chrono::high_resolution_clock::now();

    // Print timings
    std::cout << "GPU time for graph (" << graph.width() << ", " << graph.height() << "):"
              << "\n - Upload time: "
              << std::chrono::duration<double>(uploadStop - uploadStart).count() << " seconds"
              << "\n - Kernels runtime: "
              << std::chrono::duration<double>(kernelsStop - kernelsStart).count() << " seconds"
              << "\n - Download time: "
              << std::chrono::duration<double>(downloadStop - downloadStart).count() << " seconds"
              << std::endl;

    return {}; // TODO
}
