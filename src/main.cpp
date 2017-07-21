#include "Graph.h"
#include "astar.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

static float costs(const std::vector<Node> &path) {
    if (path.empty())
        return 0.0f;

    const auto &graph = path.front().graph();
    auto        node = path.begin();
    float       costs = 0.0f;

    for (auto pred = node++; node != path.end(); pred = node++)
        costs += graph.pathCost(*pred, *node);

    return costs;
}

int main() {
    // Generate graph and obstacles
    Graph graph(100, 100);
    graph.generateObstacles();

    // Generate source/destination pairs
    const int pathCount = 100;

    std::random_device                 rd;
    std::default_random_engine         generator(rd());
    std::uniform_int_distribution<int> distX(0, graph.width() - 1);
    std::uniform_int_distribution<int> distY(0, graph.height() - 1);

    std::vector<std::pair<Position, Position>> srcDstList;
    for (int i = 0; i < pathCount; ++i)
        srcDstList.emplace_back(Position{distX(generator), distY(generator)},
                                Position{distX(generator), distY(generator)});

    // CPU (reference) runs
    std::vector<std::vector<Node>> cpuPaths;
    cpuPaths.reserve(srcDstList.size());

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (const auto &srcDst : srcDstList)
        cpuPaths.emplace_back(cpuAStar(graph, srcDst.first, srcDst.second));
    const auto cpuStop = std::chrono::high_resolution_clock::now();

    // Print cpu timing
    std::cout << "CPU time for " << pathCount
              << " runs: " << std::chrono::duration<double>(cpuStop - cpuStart).count()
              << " seconds" << std::endl;

    // Print graph (with first path) to image
    graph.toPfm("cpuAStar.pfm", cpuPaths.front());

    try {
        // GPU runs
        const auto gpuPaths = gpuAStar(graph, srcDstList);

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
        graph.toPfm("gpuAStar.pfm", gpuPaths.front());
    } catch (std::exception &e) {
        std::cerr << "A* execution failed:\n" << e.what() << std::endl;
    }

#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::flush;
    std::cin.ignore();
#endif
    return 0;
}
