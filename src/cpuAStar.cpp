#include "astar.h"

#include "PriorityQueue.h"
#include <map>

namespace {
struct NodeCost {
    NodeCost(Node _node, float _totalCost, float _heuristic, Node _predecessor)
        : node(std::move(_node)), totalCost(_totalCost), heuristic(_heuristic),
          predecessor(std::move(_predecessor)) {}

    Node  node;
    float totalCost;   // g-value
    float heuristic;   // h-value
    Node  predecessor; // to recreate path
};

// Comparator to put cheapest nodes first.
struct Compare {
    bool operator()(const NodeCost &a, const NodeCost &b) {
        return a.totalCost + a.heuristic > b.totalCost + b.heuristic;
    }
};
} // namespace

// Implementation like in https://de.wikipedia.org/wiki/A*-Algorithmus#Funktionsweise
std::vector<Node> cpuAStar(const Graph &graph, const Position &source,
                           const Position &destination) {
	if (source == destination)
		return { {{graph, source}, {graph, destination}} };

    // Open and closed list
    PriorityQueue<NodeCost, Compare> open;
    // (Tree) map seems to perform _much_ better than the unordered hash map!
    std::map<Node, Node> closed; // store predecessor as value to recreate path

    // Begin at source
    const Node sourceNode(graph, source);
    open.emplace(sourceNode, 0.0f, 0.0f, sourceNode);

    while (!open.empty()) {
        const auto current = open.top();
        open.pop();

        // Reached destination! Restore path and return.
        if (current.node.position() == destination) {
            std::vector<Node> result = {current.node, current.predecessor};

            for (auto it = closed.find(result.back()); it->second != result.back();
                 it = closed.find(result.back())) {
                result.push_back(it->second);
            }

            std::reverse(result.begin(), result.end());

            return result;
        }

        closed.emplace(current.node, current.predecessor);

        // Expand node
        for (const auto &neighbor : current.node.neighbors()) {
            const auto &nbNode = neighbor.first;
            const auto &nbStepCost = neighbor.second;

            // Already visited (cycle)
            if (closed.count(nbNode) != 0)
                continue;

            const auto nbTotalCost = current.totalCost + nbStepCost;
            const auto nbIndex =
                open.find_if([&](const NodeCost &nc) { return nc.node == nbNode; });

            // Node already queued for visiting and other path cost is equal or better
            if (nbIndex < open.size() && open[nbIndex].totalCost <= nbTotalCost)
                continue;

            const auto nbHeuristic = (destination - nbNode.position()).length();

            if (nbIndex < open.size())
                open.update(nbIndex, {nbNode, nbTotalCost, nbHeuristic, current.node});
            else
                open.emplace(nbNode, nbTotalCost, nbHeuristic, current.node);
        }
    }

    // No path found
    return {};
}
