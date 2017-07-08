#include "Node.h"

Node::Node(const Graph &g, Position p) : m_graph(&g), m_position(std::move(p)) {}

std::vector<std::pair<Node, float>> Node::neighbors() const {
    std::vector<std::pair<Node, float>> neighbors;
    neighbors.reserve(4);

    // clockwise
    if (m_position.y > 0) {
        // top neighbor
        Node neighbor(*m_graph, {m_position.x, m_position.y - 1});
        auto cost = m_graph->pathCost(m_position, neighbor.position());
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.x < m_graph->width()) {
        // right neighbor
        Node neighbor(*m_graph, {m_position.x + 1, m_position.y});
        auto cost = m_graph->pathCost(m_position, neighbor.position());
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.y < m_graph->height()) {
        // bottom neighbor
        Node neighbor(*m_graph, {m_position.x, m_position.y + 1});
        auto cost = m_graph->pathCost(m_position, neighbor.position());
        neighbors.emplace_back(neighbor, cost);
    }
    if (m_position.x > 0) {
        // left neighbor
        Node neighbor(*m_graph, {m_position.x - 1, m_position.y});
        auto cost = m_graph->pathCost(m_position, neighbor.position());
        neighbors.emplace_back(neighbor, cost);
    }

    return neighbors;
}
