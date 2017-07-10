#include "Graph.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>

Graph::Graph(int width, int height) : m_width(width), m_height(height) {
    // Set default cost for each node to 1.0f
    m_costs.resize(width * height, 1.0f);
}

void Graph::generateObstacles(int amount) {
    std::random_device         rd;
    std::default_random_engine generator(rd());

    std::normal_distribution<float> distX(m_width / 2, m_width / 4);
    std::normal_distribution<float> distY(m_height / 2, m_height / 4);

    for (int i = 0; i < amount; ++i) {
        const Position center = {(int) std::round(distX(generator)),
                                 (int) std::round(distY(generator))};
        const int radius = std::min(m_width, m_height) / 10;

        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                const Position current = {center.x + x, center.y + y};

                const bool inBounds =
                    current.x >= 0 && current.x < m_width && current.y >= 0 && current.y < m_height;
                if (!inBounds)
                    continue;

                const float distance = (current - center).length();
                if (distance < (float) radius)
                    m_costs[current.y * m_width + current.x] +=
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

float Graph::pathCost(const Position &source, const Position &destination) const {
    assert(source.x >= 0 && source.x < m_width);
    assert(source.y >= 0 && source.y < m_height);
    assert(destination.x >= 0 && destination.x < m_width);
    assert(destination.y >= 0 && destination.y < m_height);

    // This function is only legal for neighbors!
    assert((std::abs(source.x - destination.x) == 1 && source.y == destination.y) ||
           (std::abs(source.y - destination.y) == 1 && source.x == destination.x));

    return std::max(m_costs[source.y * m_width + source.x],
                    m_costs[destination.y * m_width + destination.x]);
}
