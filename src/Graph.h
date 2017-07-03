#pragma once

#include <string>

class Graph {
public:
    Graph(int width, int height);

    void toPfm(const std::string &filePath) const;

private:
    int m_width;
    int m_height;
};
