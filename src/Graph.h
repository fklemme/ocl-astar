#pragma once

#include "Position.h"
#include <string>
#include <vector>

class Graph {
public:
    Graph(int width, int height);

    float cost(const Position &source, const Position &destination) const;

    void toPfm(const std::string &filePath) const;

    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    int                m_width;
    int                m_height;
    std::vector<float> m_costs;
};
