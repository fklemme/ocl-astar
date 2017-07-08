#pragma once

#include "Position.h"
#include <vector>

class Graph;

class Node {
public:
    Node(const Graph &g, Position p);

    std::vector<std::pair<Node, float>> neighbors() const;

    const Position &position() const { return m_position; }

private:
    const Graph *m_graph;
    Position     m_position;
};

// Ignore graph here...
inline bool operator==(const Node &a, const Node &b) { return a.position() == b.position(); }
inline bool operator!=(const Node &a, const Node &b) { return a.position() != b.position(); }

// Needed for set/map
inline bool operator<(const Node &a, const Node &b) {
    const auto &pa = a.position();
    const auto &pb = b.position();
    return pa.y < pb.y || (pa.y == pb.y && pa.x < pb.x);
}
