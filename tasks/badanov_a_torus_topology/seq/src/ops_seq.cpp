#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"

namespace badanov_a_torus_topology {

BadanovATorusTopologySEQ::BadanovATorusTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovATorusTopologySEQ::ValidationImpl() {
  const auto &in = GetInput();
  return !std::get<2>(in).empty();
}

bool BadanovATorusTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

TorusCoords BadanovATorusTopologySEQ::RankToCoords(int rank, int grid_size) {
  TorusCoords coords{};
  coords.rank = rank;
  coords.x = rank % grid_size;
  coords.y = rank / grid_size;
  return coords;
}

int BadanovATorusTopologySEQ::CoordsToRank(int x, int y, int grid_size) {
  x = (x % grid_size + grid_size) % grid_size;
  y = (y % grid_size + grid_size) % grid_size;
  return (y * grid_size) + x;
}

double BadanovATorusTopologySEQ::CalculateTorusDistance(const TorusCoords &src, const TorusCoords &dst, int grid_size) {
  int dx = std::abs(dst.x - src.x);
  int dy = std::abs(dst.y - src.y);

  dx = std::min(dx, grid_size - dx);
  dy = std::min(dy, grid_size - dy);

  return std::sqrt(static_cast<double>((dx * dx) + (dy * dy)));
}

bool BadanovATorusTopologySEQ::RunImpl() {
  const auto &in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto &data = std::get<2>(in);

  std::vector<double> result;

  if (src == dst) {
    result = data;
  } else {
    const int grid_size = 10;
    const int virtual_size = grid_size * grid_size;

    int src_rank = static_cast<int>(src) % virtual_size;
    int dst_rank = static_cast<int>(dst) % virtual_size;

    TorusCoords src_coords = RankToCoords(src_rank, grid_size);
    TorusCoords dst_coords = RankToCoords(dst_rank, grid_size);

    double distance = CalculateTorusDistance(src_coords, dst_coords, grid_size);

    result = data;
    double scale = 1.0 / (1.0 + distance);
    for (auto &val : result) {
      val *= scale;
    }
  }

  GetOutput() = result;
  return true;
}

bool BadanovATorusTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_torus_topology
