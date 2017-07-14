#include "astar.h"

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t', possible
                                // loss of data
#include <boost/compute.hpp>
#pragma warning(pop)

#include <iostream> // DEBUG

std::vector<Node> gpuAStar(const Graph & /*g*/, const Position & /*source*/,
                           const Position & /*destination*/) {
    namespace ocl = boost::compute;

    ocl::device gpu = ocl::system::default_device();
    std::cout << "Selected device: " << gpu.name() << std::endl;

    ocl::context       ctx(gpu);
    ocl::command_queue queue(ctx, gpu);

    // TODO...

    return {};
}
