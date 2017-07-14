#include "astar.h"

#include <boost/compute.hpp>

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
