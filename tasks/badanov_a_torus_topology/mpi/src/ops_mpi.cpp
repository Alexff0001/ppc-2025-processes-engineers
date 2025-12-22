#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"

namespace badanov_a_torus_topology {

BadanovATorusTopologyMPI::BadanovATorusTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovATorusTopologyMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const auto &in = GetInput();
  const int src = static_cast<int>(std::get<0>(in));
  const int dst = static_cast<int>(std::get<1>(in));
  return src >= 0 && dst >= 0 && src < world_size && dst < world_size;
}

bool BadanovATorusTopologyMPI::PreProcessingImpl() {
  return true;
}

TorusCoords BadanovATorusTopologyMPI::RankToCoords(int rank, int rows, int cols) {
  (void)rows;
  TorusCoords coords{};
  coords.rank = rank;
  coords.x = rank % cols;
  coords.y = rank / cols;
  return coords;
}

int BadanovATorusTopologyMPI::CoordsToRank(int x, int y, int rows, int cols) {
  x = (x % cols + cols) % cols;
  y = (y % rows + rows) % rows;
  return (y * cols) + x;
}

int BadanovATorusTopologyMPI::CalculateStepDelta(int delta, int dimension_size) {
  if (delta > dimension_size / 2) {
    delta -= dimension_size;
  } else if (delta < -dimension_size / 2) {
    delta += dimension_size;
  }
  return delta;
}

std::vector<int> BadanovATorusTopologyMPI::GetRoute(int src_rank, int dst_rank, int rows, int cols) {
  std::vector<int> route;

  if (src_rank == dst_rank) {
    route.push_back(src_rank);
    return route;
  }

  TorusCoords src_coords = RankToCoords(src_rank, rows, cols);
  TorusCoords dst_coords = RankToCoords(dst_rank, rows, cols);

  int dx = CalculateStepDelta(dst_coords.x - src_coords.x, cols);
  int dy = CalculateStepDelta(dst_coords.y - src_coords.y, rows);

  int current_x = src_coords.x;
  int current_y = src_coords.y;

  route.push_back(src_rank);

  int x_step = (dx > 0) ? 1 : -1;
  for (int i = 0; i < std::abs(dx); i++) {
    current_x += x_step;
    current_x = (current_x < 0) ? cols - 1 : current_x;
    current_x = (current_x >= cols) ? 0 : current_x;

    route.push_back(CoordsToRank(current_x, current_y, rows, cols));
  }

  int y_step = (dy > 0) ? 1 : -1;
  for (int i = 0; i < std::abs(dy); i++) {
    current_y += y_step;
    current_y = (current_y < 0) ? cols - 1 : current_y;
    current_y = (current_y >= cols) ? 0 : current_y;

    route.push_back(CoordsToRank(current_x, current_y, rows, cols));
  }

  return route;
}

bool BadanovATorusTopologyMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto &data = std::get<2>(in);

  auto &out = GetOutput();
  out.clear();

  const int src_rank = static_cast<int>(src);
  const int dst_rank = static_cast<int>(dst);

  if (src_rank == dst_rank) {
    if (world_rank == src_rank) {
      out = data;
    }
    return true;
  }

  if (world_size <= 1 || src_rank < 0 || dst_rank < 0 || 
      src_rank >= world_size || dst_rank >= world_size) {
    return true;
  }

  int rows = 1;
  int cols = world_size;
  
  if (world_size > 1) {
    rows = static_cast<int>(std::sqrt(world_size));
    while (rows > 0 && world_size % rows != 0) {
      rows--;
    }
    cols = world_size / rows;
  }

  std::vector<int> route = GetRoute(src_rank, dst_rank, rows, cols);
  
  int position_in_route = -1;
  for (size_t i = 0; i < route.size(); ++i) {
    if (world_rank == route[i]) {
      position_in_route = static_cast<int>(i);
      break;
    }
  }

  if (position_in_route < 0) {
    return true;
  }

  const int tag_data = 0;

  if (position_in_route == 0) {
    if (route.size() > 1) {
      int next_hop = route[1];
      MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, next_hop, tag_data, MPI_COMM_WORLD);
    }
  } else if (position_in_route == static_cast<int>(route.size()) - 1) {
    out.resize(data.size());
    MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    std::vector<double> buffer(data.size());
    MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    int next_hop = route[position_in_route + 1];
    MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, next_hop, tag_data, MPI_COMM_WORLD);
  }

  return true;
}

bool BadanovATorusTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_torus_topology
