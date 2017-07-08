#include "Graph.h"

#include <cassert>
#include <fstream>
#include <iomanip>

Graph::Graph(int width, int height) : m_width(width), m_height(height) {
    // Set default cost for each neighbor to 1.0f
    m_costs.resize(2 * width * height, 1.0f);
    // Times two: Save "cost to the right, cost to the bottom" per node.
    // Allocated a little bit too much memory for access simplicity.
}

float Graph::pathCost(const Position &source, const Position &destination) const {
    assert(source.x >= 0 && source.x < m_width);
    assert(source.y >= 0 && source.y < m_height);
    assert(destination.x >= 0 && destination.x < m_width);
    assert(destination.y >= 0 && destination.y < m_height);

    // This function is only legal for neighbors!
    assert((destination - source).length() == 1.0f);

    const int x = (source.x + destination.x) / 2;
    const int y = (source.y + destination.y) / 2;
    const int index = 2 * (y * m_width + x);

    return m_costs[source.x != destination.x ? index : index + 1];
}

void Graph::toPfm(const std::string &filePath) const {
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

    for (int row = m_height - 1; row >= 0; --row) {
        for (int col = 0; col < m_width; ++col) {

            float costs = 0.0f;
            costs += row > 0 ? m_costs[2 * ((row - 1) * m_width + col) + 1] : 1.0f;
            costs += col < m_width - 1 ? m_costs[2 * (row * m_width + col)] : 1.0f;
            costs += row < m_height - 1 ? m_costs[2 * (row * m_width + col) + 1] : 1.0f;
            costs += col > 0 ? m_costs[2 * (row * m_width + (col - 1))] : 1.0f;

            raster.emplace_back(0.0f, 0.0f, 4.0f / costs);
        }
    }

    assert(raster.size() == m_width * m_height);
    out.write(reinterpret_cast<const char *>(raster.data()), raster.size() * sizeof(RGB));
}
