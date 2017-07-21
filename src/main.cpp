#include "Graph.h"
#include "astar.h"
#include <algorithm>
#include <iostream>

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
    Graph graph(200, 200);
    graph.generateObstacles();

    const Position start{10, 20};
    const Position destination{180, 190};

    // CPU (reference) run
    const auto cpuPath = cpuAStar(graph, start, destination);
    graph.toPfm("cpuAStar.pfm", cpuPath);

    try {
        // GPU run
        const auto  gpuPaths = gpuAStar(graph, {{start, destination}});
        const auto &gpuPath = gpuPaths.front(); // path from first agent

        if (std::equal(cpuPath.begin(), cpuPath.end(), gpuPath.begin(), gpuPath.end())) {
            std::cout << "GPU A*: Gold test passed! (exact match)" << std::endl;
        } else if (cpuPath.size() == gpuPath.size() && std::abs(costs(cpuPath) - costs(gpuPath)) < 0.1f) {
            std::cout << "GPU A*: Gold test passed! (equal match)" << std::endl;
        } else {
            std::cerr << "GPU A*: Gold test failed!"
                      << "\nPath length CPU: " << cpuPath.size() << ", GPU: " << gpuPath.size()
                      << "\nPath cost CPU: " << costs(cpuPath) << ", GPU: " << costs(gpuPath)
                      << std::endl;
        }

        // Print graph to image
        graph.toPfm("gpuAStar.pfm", gpuPath);
    } catch (std::exception &e) {
        std::cerr << "A* execution failed:\n" << e.what() << std::endl;
    }

#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::flush;
    std::cin.ignore();
#endif
    return 0;
}
