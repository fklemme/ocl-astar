#pragma once

#include <cmath>

class Position {
public:
    Position operator-(const Position &other) const { return {x - other.x, y - other.y}; }

    float length() const { return (float) std::sqrt(x * x + y * y); }

    int x;
    int y;
};

inline bool operator==(const Position &a, const Position &b) { return a.x == b.x && a.y == b.y; }
inline bool operator!=(const Position &a, const Position &b) { return a.x != b.x || a.y != b.y; }
