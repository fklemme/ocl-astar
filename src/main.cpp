#include "Graph.h"
#include "astar.h"
#include <iostream>

int main() {
    // Generate graph and obstacles
    Graph g(800, 600);
    g.generateObstacles();

    // Find path through graph
    const Position start = {10, 10};
    const Position destination = {790, 590};
    const auto     path = cpuAStar(g, start, destination);

    // Print graph to image
    g.toPfm("graph.pfm", path);

#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::endl;
    std::cin.ignore();
#endif

    return 0;
}
