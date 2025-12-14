#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace badanov_a_torus_topology {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace badanov_a_torus_topology
