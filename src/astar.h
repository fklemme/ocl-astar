#pragma once

// Workaround for my old crappy notebook: Force OpenCL 1.1!
// For some reason, OpenCL 1.2 is declared in the header, although not supported on the platform.
#if 0
#include <boost/compute/cl.hpp>
#undef CL_VERSION_1_2
#endif

#include "Graph.h"
#include "Node.h"
#include "Position.h"
#include <vector>

#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t',
// possible loss of data
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>
#pragma warning(pop)

// Enable printing of debug information from functions below.
#define DEBUG_OUTPUT

std::vector<Node> cpuAStar(const Graph &graph, const Position &source, const Position &destination);

std::vector<std::vector<Node>>
gpuAStar(const Graph &graph, const std::vector<std::pair<Position, Position>> &srcDstList,
         const boost::compute::device &clDevice = boost::compute::system::default_device());

std::vector<Node>
gpuGAStar(const Graph &graph, const Position &source, const Position &destination,
          const boost::compute::device &clDevice = boost::compute::system::default_device());
