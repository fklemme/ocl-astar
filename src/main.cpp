#include "Graph.h"
#include "astar.h"
#include <iostream>

int main() {
    const Graph g(800, 600);

    const auto path = cpuAStar(g, {10, 10}, {790, 590});

    g.toPfm("graph.pfm", path);

    for (const auto &node : path)
        std::cout << "(" << node.position().x << ", " << node.position().y << ") ";
    std::cout << std::endl;

#ifdef _WIN32
    std::cout << "\nPress ENTER to continue..." << std::endl;
    std::cin.ignore();
#endif

    return 0;
}
