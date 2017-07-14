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

std::vector<Node> gpuAStar(const Graph &graph, const Position & /*source*/,
                           const Position & /*destination*/) {
    namespace compute = boost::compute;

    // Set up OpenCL environment and build program
    compute::device gpu = compute::system::default_device();
    std::cout << "Selected device: " << gpu.name() << std::endl;

    compute::context       context(gpu);
    compute::command_queue queue(context, gpu);

    auto program = compute::program::create_with_source_file("src/gpuAStar.cl", context);
    program.build();

    // Set up data structures
    static_assert(sizeof(compute::float4_) == 16,
                  "Nodes: represented by four floats (total of 16 bytes)");
    std::vector<compute::float4_>     h_nodes(graph.size());
    compute::vector<compute::float4_> d_nodes(graph.size(), context);

    int index = 0;
    for (int y = 0; y < graph.height(); ++y) {
        for (int x = 0; x < graph.width(); ++x) {
            h_nodes[index] = compute::float4_((float) index, (float) x, (float) y, 0.0f);
            ++index;
        }
    }

    // Create kernel
    compute::kernel kernel(program, "gpuAStar");
    kernel.set_arg(0, d_nodes);

    // Upload data
    compute::copy(h_nodes.begin(), h_nodes.end(), d_nodes.begin(), queue);

    // Run kernel
    std::size_t globalWorkSize = 4, localWorkSize = 0;
    queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);

    return {};
}
