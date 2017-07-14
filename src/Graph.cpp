#include "Graph.h"

#include "Node.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'double' to 'float', possible loss of data
#include <random>
#pragma warning(pop)

Graph::Graph(int width, int height) : m_width(width), m_height(height) {
    // Set default cost for each node to 1.0f
    m_costs.resize(width * height, 1.0f);
}

void Graph::generateObstacles(int amount) {
    std::random_device         rd;
    std::default_random_engine generator(rd());

    std::normal_distribution<float> distX(m_width / 2.0f, m_width / 4.0f);
    std::normal_distribution<float> distY(m_height / 2.0f, m_height / 4.0f);

    for (int i = 0; i < amount; ++i) {
        const Position center = {(int) std::round(distX(generator)),
                                 (int) std::round(distY(generator))};
        const int      radius = std::min(m_width, m_height) / 10;

        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                const Node current{*this, {center.x + x, center.y + y}};

                if (!current.inBounds())
                    continue;

                const float distance = (current.position() - center).length();
                if (distance < (float) radius)
                    m_costs[current.position().y * m_width + current.position().x] +=
                        std::sqrt(radius * radius - distance * distance) / 10;
            }
        }
    }
}

void Graph::toPfm(const std::string &filePath, const std::vector<Node> &path) const {
    // http://netpbm.sourceforge.net/doc/pfm.html
    std::ofstream out(filePath, std::ios::trunc);
    out << "PF\n";                             // Identifier Line
    out << m_width << ' ' << m_height << '\n'; // Dimensions Line
    out << std::fixed << -1.0f << '\n';        // Scale Factor / Endianness

    struct RGB {
        RGB(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
        float r, g, b;
    };

    static_assert(sizeof(float) == 4, "float type is not 4 bytes!");
    static_assert(sizeof(RGB) == 12, "RGB type is not 12 bytes!");

    // PFM raster
    std::vector<RGB> raster;
    raster.reserve(m_width * m_height);

    // Draw graph
    for (int row = m_height - 1; row >= 0; --row) {
        for (int col = 0; col < m_width; ++col) {
            const float brightness = 1.0f / m_costs[row * m_width + col];
            raster.emplace_back(0.0f, brightness, brightness);
        }
    }

    // Draw path
    for (const auto &node : path) {
        auto &pixel = raster[(m_height - node.position().y - 1) * m_width + node.position().x];
        pixel.r = 1.0f;
        pixel.g /= 10;
        pixel.b /= 10;
    }

    out.write(reinterpret_cast<const char *>(raster.data()), raster.size() * sizeof(RGB));
}

float Graph::pathCost(const Node &source, const Node &destination) const {
    assert(source.inBounds());
    assert(destination.inBounds());

    const auto &src = source.position();
    const auto &dst = destination.position();

// This function is only legal for neighbors!
#ifndef DIAGONAL
    // Orthogonal only
    assert((std::abs(src.x - dst.x) == 1 && src.y == dst.y) ||
           (std::abs(src.y - dst.y) == 1 && src.x == dst.x));

    return std::max(m_costs[src.y * m_width + src.x], m_costs[dst.y * m_width + dst.x]);
#else
    // Diagonal connections as well
    assert(std::abs(src.x - dst.x) <= 1 && std::abs(src.y - dst.y) <= 1);

    const bool diagonal = src.x != dst.x && src.y != dst.y;
    const auto cost = std::max(m_costs[src.y * m_width + src.x], m_costs[dst.y * m_width + dst.x]);
    constexpr float sqrt2 = 1.41421356237f;
    return diagonal ? sqrt2 * cost : cost;
#endif
}
