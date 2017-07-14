#include "Graph.h"
#include "astar.h"
#include <algorithm>
#include <iostream>

int main() {
    // Generate graph and obstacles
    Graph g(800, 600);
    g.generateObstacles();

    // Find path through graph
    const Position start = {10, 10};
    const Position destination = {790, 590};
    const auto     cpuPath = cpuAStar(g, start, destination);
    const auto     gpuPath = gpuAStar(g, start, destination);

    if (std::equal(cpuPath.begin(), cpuPath.end(), gpuPath.begin(), gpuPath.end())) {
        std::cout << "GPU A*: Gold test passed!" << std::endl;
    } else {
        std::cerr << "GPU A*: Gold test failed!" << std::endl;
    }

    // Print graph to image
    g.toPfm("cpuAStar.pfm", cpuPath);
    g.toPfm("gpuAStar.pfm", gpuPath);

    // Done
#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::endl;
    std::cin.ignore();
#endif
    return 0;
}
