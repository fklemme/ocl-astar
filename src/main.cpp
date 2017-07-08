#include "Graph.h"
#include "astar.h"
#include <iostream>

int main() {
    Graph g(200, 200);

    auto path = cpuAStar(g, {10, 20}, {190, 180});

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
