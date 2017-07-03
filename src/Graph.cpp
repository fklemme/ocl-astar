#include "Graph.h"

#include <fstream>
#include <iomanip>

Graph::Graph(int width, int height) : m_width(width), m_height(height) {}

void Graph::toPfm(const std::string &filePath) const {
    // http://netpbm.sourceforge.net/doc/pfm.html
    std::ofstream out(filePath, std::ios::trunc);
    out << "Pf\n"; // Identifier Line
    out << m_width << ' ' << m_height << '\n'; // Dimensions Line
    out << std::fixed << -1.0f << '\n'; // Scale Factor / Endianness

    // PFM raster
    static_assert(sizeof(float) == 4);
    for (int row = m_height - 1; row >= 0; --row) {
        for (int col = 0; col < m_width; ++col) {
            float value= (row * col) /10000.0f;
            value = value > 1.0f ? 1.0f : value;
            out.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }
    }
}
