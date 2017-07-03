#include "Graph.h"
#include "astar.h"
#include <iostream>

int main() {
    Graph g(200, 200);

    g.toPfm("graph.pfm");

    auto path = cpuAStar(g, {10, 10}, {190, 190});

    return 0;
}
