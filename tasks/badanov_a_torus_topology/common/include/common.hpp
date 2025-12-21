#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace badanov_a_torus_topology {

using InType = std::tuple<size_t, size_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<size_t, size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct TorusCoords {
  int x;
  int y;
  int rank;
};

}  // namespace badanov_a_torus_topology
