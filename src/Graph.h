#pragma once

#include "Node.h"
#include "Position.h"
#include <string>
#include <vector>

class Graph {
public:
    Graph(int width, int height);

    void generateObstacles(int amount = 10);
    void toPfm(const std::string &filePath, const std::vector<Node> &path = {}) const;

    float pathCost(const Position &source, const Position &destination) const;

    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    std::vector<float *> pathsAroundNode(const Position &nodePosition);

    int                m_width;
    int                m_height;
    std::vector<float> m_costs;
};
