#include "astar.h"

#include <algorithm>
#include <map>
#include <vector>

namespace {
struct NodeCost {
    NodeCost(Node _node, float _cost, float _heuristic, Node _predecessor)
        : node(std::move(_node)), cost(_cost), heuristic(_heuristic),
          predecessor(std::move(_predecessor)) {}

    Node  node;
    float cost;        // g-value
    float heuristic;   // h-value
    Node  predecessor; // to recreate path
};

// Comparator to put cheapest nodes first.
bool compare(const NodeCost &a, const NodeCost &b) {
    return a.cost + a.heuristic > b.cost + b.heuristic;
};

class NodePrioQueue {
public:
    NodeCost top() const { return m_heap.front(); }
    bool     empty() const { return m_heap.empty(); }
    auto     size() const { return m_heap.size(); }

    void push(const NodeCost &nc) {
        m_heap.push_back(nc);
        std::push_heap(m_heap.begin(), m_heap.end(), compare);
    }

    template <typename... Args>
    void emplace(Args &&... args) {
        m_heap.emplace_back(std::forward<Args>(args)...);
        std::push_heap(m_heap.begin(), m_heap.end(), compare);
    }

    NodeCost pop() {
        std::pop_heap(m_heap.begin(), m_heap.end(), compare);
        auto result = std::move(m_heap.back());
        m_heap.pop_back();
        return result;
    }

    // This member function doesn't exist in std::priority_queue.
    template <typename Function>
    bool findNodeAndApply(const Node &node, Function f) {
        int index = findNode(node);

        if (index < (int) m_heap.size()) {
            auto old = m_heap[index];
            f(m_heap[index]);

            // If node has changed, update heap.
            if (compare(m_heap[index], old) == compare(old, m_heap[index])) {
                std::make_heap(m_heap.begin(), m_heap.end(), compare); // probably suboptimal
            }

            return true;
        }
        return false;
    }

private:
    int findNode(const Node &node, int index = 0) const {
        if (index >= (int) m_heap.size())
            return m_heap.size();

        if (m_heap[index].node == node)
            return index;

        return std::min(findNode(node, index * 2 + 1), findNode(node, index * 2 + 2));
    }

    std::vector<NodeCost> m_heap;
};
}

std::vector<Node> cpuAStar(const Graph &g, const Position &source, const Position &destination) {

    // Open and closed list
    NodePrioQueue open;
    std::map<Node, Node> closed; // store predecessor to recreate path

    // Begin at source
    const Node sourceNode(g, source);
    open.emplace(sourceNode, 0.0f, 0.0f, sourceNode);

    while (!open.empty()) {
        const auto current = open.top();
        open.pop();

        if (current.node.position() == destination) {
            return {}; // TODO
        }

        closed.emplace(current.node, current.predecessor);

        // Expand node
        for (const auto &neighbor : current.node.neighbors()) {
            const auto &neighborNode = neighbor.first;
            const auto &neighborCost = neighbor.second;

            if (closed.count(neighborNode) == 0) {
                const auto cost = current.cost + neighborCost;
                const auto heuristic = (destination - neighborNode.position()).length();

                bool nodeAlreadyInOpenList = open.findNodeAndApply(neighborNode, [&](NodeCost &nc) {
                    if (cost < nc.cost) {
                        // Found a cheaper path. Update node.
                        nc.cost = cost;
                        nc.predecessor = current.node;
                    }
                });

                if (!nodeAlreadyInOpenList) {
                    open.emplace(neighborNode, cost, heuristic, current.node);
                }
            }
        }
    }

    // No path found
    return {};
}
