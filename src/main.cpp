#include "Graph.h"
#include "astar.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace compute = boost::compute;

// Little helper for validation
static float costs(const std::vector<Node> &path) {
    if (path.empty())
        return 0.0f;

    auto &graph = path.front().graph();
    auto  node = path.begin();
    float costs = 0.0f;

    for (auto pred = node++; node != path.end(); pred = node++)
        costs += graph.pathCost(*pred, *node);

    return costs;
}

// Run multi-agent A*
static void runAStar(const compute::device &clDevice) {
    // Generate graph and obstacles
    Graph graph(50, 50); // should be small
    graph.generateObstacles();

    // Generate source/destination pairs
    const int                          pathCount = 500; // should be big
    std::random_device                 rd;
    std::default_random_engine         generator(rd());
    std::uniform_int_distribution<int> distX(0, graph.width() - 1);
    std::uniform_int_distribution<int> distY(0, graph.height() - 1);

    std::vector<std::pair<Position, Position>> srcDstList;
    for (int i = 0; i < pathCount; ++i)
        srcDstList.emplace_back(Position{distX(generator), distY(generator)},
                                Position{distX(generator), distY(generator)});

    // CPU reference run
    std::vector<std::vector<Node>> cpuPaths;
    cpuPaths.reserve(srcDstList.size());

    std::cout << " ----- CPU reference run..." << std::endl;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (const auto &srcDst : srcDstList)
        cpuPaths.emplace_back(cpuAStar(graph, srcDst.first, srcDst.second));
    const auto cpuStop = std::chrono::high_resolution_clock::now();

    // Print cpu timing
    std::cout << "CPU time for " << pathCount
              << " runs: " << std::chrono::duration<double>(cpuStop - cpuStart).count()
              << " seconds" << std::endl;

    // Print graph (with first path) to image
    graph.toPfm("AStarCPU.pfm", cpuPaths.front());

    try {
        // GPU A* run
        std::cout << " ----- GPU A* run..." << std::endl;
        const auto gpuPaths = gpuAStar(graph, srcDstList, clDevice);

        assert(cpuPaths.size() == gpuPaths.size());

        for (std::size_t i = 0; i < cpuPaths.size(); ++i) {
            const auto &cpuPath = cpuPaths[i];
            const auto &gpuPath = gpuPaths[i];

            if (std::equal(cpuPath.begin(), cpuPath.end(), gpuPath.begin(), gpuPath.end())) {
                // std::cout << "GPU A* " << i << ": Gold test passed! (exact match)" << std::endl;
            } else if (cpuPath.size() == gpuPath.size() &&
                       std::abs(costs(cpuPath) - costs(gpuPath)) < 0.1f) {
                // std::cout << "GPU A* " << i << ": Gold test passed! (equal match)" << std::endl;
            } else {
                std::cerr << "GPU A* " << i << ": Gold test failed!"
                          << "\n - Path length CPU: " << cpuPath.size()
                          << ", GPU: " << gpuPath.size() << "\n - Path cost CPU: " << costs(cpuPath)
                          << ", GPU: " << costs(gpuPath) << std::endl;
            }
        }

        // Print graph (with first path) to image
        graph.toPfm("AStarGPU.pfm", gpuPaths.front());
    } catch (std::exception &e) {
        std::cerr << "A* execution failed:\n" << e.what() << std::endl;
    }
}

// Run parallel GA*
static void runGAStar(const compute::device &clDevice) {
    // Generate graph and obstacles
    Graph graph(200, 200); // should be big
    graph.generateObstacles();

    const Position source{10, 20};
    const Position destination{graph.width() - 10, graph.height() - 20};

    // CPU reference run
    std::cout << " ----- CPU reference run..." << std::endl;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    const auto cpuPath = cpuAStar(graph, source, destination);
    const auto cpuStop = std::chrono::high_resolution_clock::now();

    // Print cpu timing
    std::cout << "CPU time for graph (" << graph.width() << ", " << graph.height()
              << "): " << std::chrono::duration<double>(cpuStop - cpuStart).count() << " seconds"
              << std::endl;

    // Print graph (with first path) to image
    graph.toPfm("GAStarCPU.pfm", cpuPath);

    try {
        // GPU GA* run
        std::cout << " ----- GPU GA* run..." << std::endl;
        const auto gpuPath = gpuGAStar(graph, source, destination, clDevice);

        if (std::equal(cpuPath.begin(), cpuPath.end(), gpuPath.begin(), gpuPath.end())) {
            // std::cout << "GPU GA* " << i << ": Gold test passed! (exact match)" << std::endl;
        } else if (cpuPath.size() == gpuPath.size() &&
                   std::abs(costs(cpuPath) - costs(gpuPath)) < 0.1f) {
            // std::cout << "GPU GA* " << i << ": Gold test passed! (equal match)" << std::endl;
        } else {
            std::cerr << "GPU GA*: Gold test failed!"
                      << "\n - Path length CPU: " << cpuPath.size() << ", GPU: " << gpuPath.size()
                      << "\n - Path cost CPU: " << costs(cpuPath) << ", GPU: " << costs(gpuPath)
                      << std::endl;
        }

        // Print graph (with first path) to image
        graph.toPfm("GAStarGPU.pfm", gpuPath);
    } catch (std::exception &e) {
        std::cerr << "GA* execution failed:\n" << e.what() << std::endl;
    }
}

int main() {
    // Select default OpenCL device
    compute::device gpu = compute::system::default_device();

    // Workaround for testing on broken AMD system: Use CPU instead.
    auto cpu = []() {
        for (auto d : compute::system::devices())
            if (d.type() == compute::device::cpu)
                return d;
        return compute::system::default_device();
    }();

    // Run multi-agent A*
    runAStar(gpu);

    // Run parallel GA*
    runGAStar(gpu);

#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::flush;
    std::cin.ignore();
#endif
    return 0;
}
