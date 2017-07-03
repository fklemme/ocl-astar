#include "astar.h"

#include <queue>
#include <set>

std::vector<Node> cpuAStar(const Graph &g, const Position &source, const Position &destination) {
    using NodeCost = std::pair<Node, float>;
    auto compare = [](const NodeCost &a, const NodeCost &b) { return a.second > b.second; };

    // Open and closed list
    std::priority_queue<NodeCost, std::vector<NodeCost>, decltype(compare)> open(compare);
    std::set<Node>    closed;
    std::vector<Node> path;

    // Begin at source
    open.emplace(Node(g, source), 0.0f);

    while (!open.empty()) {
        const auto node = open.top().first;
        const auto cost = open.top().second;
        open.pop();

        // There may be duplicates, check if already visited
        if (closed.count(node) == 1)
            continue;

        // Remember path
        path.push_back(node);
        closed.insert(node);

        if (node.position() == destination) {
            return path;
        }

        // Expand node
        for (const auto &neighbor : node.neighbors()) {
            const auto &nbNode = neighbor.first;
            const auto &nbCost = neighbor.second;

            if (closed.count(nbNode) == 0) {
                const auto totalCost = cost + nbCost + (destination - nbNode.position()).length();
                open.emplace(nbNode, totalCost);
            }
        }
    }

    // No path found
    return {};
}
