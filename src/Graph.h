#pragma once

#define GRAPH_DIAGONAL_MOVEMENT

#include <string>
#include <vector>

class Node;

class Graph {
public:
    Graph(int width, int height);

    void generateObstacles(int amount = 10);

    void toPfm(const std::string &filePath, const std::vector<Node> &path = {}) const;

    float pathCost(const Node &source, const Node &destination) const;

    int width() const { return m_width; }
    int height() const { return m_height; }
    int size() const { return m_width * m_height; }

private:
    int                m_width;
    int                m_height;
    std::vector<float> m_costs;
};
