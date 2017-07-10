#pragma once

#include "Position.h"
#include <functional>
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

// FIXME: Graph should not be ignored? For now, comparing nodes from different graphs is undefined
// behavior.
inline bool operator==(const Node &a, const Node &b) { return a.position() == b.position(); }
inline bool operator!=(const Node &a, const Node &b) { return a.position() != b.position(); }

// Needed for (tree) set/map
inline bool operator<(const Node &a, const Node &b) {
    const auto &pa = a.position();
    const auto &pb = b.position();
    return pa.y < pb.y || (pa.y == pb.y && pa.x < pb.x);
}

// Needed for unordered (hash) set/map
namespace std {
template <>
struct hash<Node> {
    using argument_type = Node;
    using result_type = size_t;

    result_type operator()(argument_type const &s) const {
        static_assert(sizeof(decltype(s.position().x)) >= 4, "Type too small for hash strategy.");

        return s.position().x ^ (s.position().y << 16);
    }
};
}
