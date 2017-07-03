#include "Graph.h"

#include <cassert>
#include <fstream>
#include <iomanip>

Graph::Graph(int width, int height) : m_width(width), m_height(height) {
    // Set default cost for each neighbor to 1.0f
    m_costs.resize((width - 1) * (height - 1), 1.0f);
}

float Graph::cost(const Position &source, const Position &destination) const {
    // This function is only legal for neighbors!
    assert((destination - source).length() == 1.0f);

    const int x = (source.x + destination.x) >> 1;
    const int y = (source.y + destination.y) >> 1;
    return m_costs[y * m_width + x];
}

void Graph::toPfm(const std::string &filePath) const {
    // http://netpbm.sourceforge.net/doc/pfm.html
    std::ofstream out(filePath, std::ios::trunc);
    out << "PF\n";                             // Identifier Line
    out << m_width << ' ' << m_height << '\n'; // Dimensions Line
    out << std::fixed << -1.0f << '\n';        // Scale Factor / Endianness

    // PFM raster
    static_assert(sizeof(float) == 4);
    for (int row = m_height - 1; row >= 0; --row) {
        for (int col = 0; col < m_width; ++col) {
            float rgb[3];
            rgb[0] = std::min((float) col / m_width, 1.0f);  // red to the right
            rgb[1] = std::min((float) row / m_height, 1.0f); // green to the bottom
            float blue = 1.33f - rgb[0] - rgb[1];            // blue spot in top-left
            rgb[2] = std::min(std::max(0.0f, blue), 1.0f);
            out.write(reinterpret_cast<const char *>(rgb), 3 * sizeof(float));
        }
    }
}
