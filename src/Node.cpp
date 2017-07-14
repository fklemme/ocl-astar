#include "Node.h"

Node::Node(const Graph &g, Position p) : m_graph(&g), m_position(std::move(p)) {}

std::vector<std::pair<Node, float>> Node::neighbors() const {
    std::vector<std::pair<Node, float>> neighbors;

#ifndef DIAGONAL
    neighbors.reserve(4);

    // clockwise
    if (m_position.y > 0) {
        // top neighbor
        const Node neighbor(*m_graph, {m_position.x, m_position.y - 1});
        const auto cost = m_graph->pathCost(*this, neighbor);
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.x < m_graph->width()) {
        // right neighbor
        const Node neighbor(*m_graph, {m_position.x + 1, m_position.y});
        const auto cost = m_graph->pathCost(*this, neighbor);
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.y < m_graph->height()) {
        // bottom neighbor
        const Node neighbor(*m_graph, {m_position.x, m_position.y + 1});
        const auto cost = m_graph->pathCost(*this, neighbor);
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.x > 0) {
        // left neighbor
        const Node neighbor(*m_graph, {m_position.x - 1, m_position.y});
        const auto cost = m_graph->pathCost(*this, neighbor);
        neighbors.emplace_back(neighbor, cost);
    }
#else
    neighbors.reserve(8);

    for (int y = m_position.y - 1; y <= m_position.y + 1; ++y) {
        for (int x = m_position.x - 1; x <= m_position.x + 1; ++x) {
            const Node neighbor{*m_graph, {x, y}};
            if (neighbor != *this && neighbor.inBounds())
                neighbors.emplace_back(neighbor, m_graph->pathCost(*this, neighbor));
        }
    }
#endif

    return neighbors;
}
