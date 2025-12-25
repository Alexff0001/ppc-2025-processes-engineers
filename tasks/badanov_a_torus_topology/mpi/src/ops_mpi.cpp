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

std::vector<int> BadanovATorusTopologyMPI::GetRoute(int src_rank, int dst_rank, int rows, int cols) {
  std::vector<int> route;

  if (src_rank == dst_rank) {
    route.push_back(src_rank);
    return route;
  }

  TorusCoords src_coords = RankToCoords(src_rank, rows, cols);
  TorusCoords dst_coords = RankToCoords(dst_rank, rows, cols);

  int dx = dst_coords.x - src_coords.x;
  int dy = dst_coords.y - src_coords.y;

  if (dx > cols / 2) {
    dx -= cols;
  } else if (dx < -cols / 2) {
    dx += cols;
  }

  if (dy > rows / 2) {
    dy -= rows;
  } else if (dy < -rows / 2) {
    dy += rows;
  }

  int current_x = src_coords.x;
  int current_y = src_coords.y;

  route.push_back(src_rank);

  while (dx != 0) {
    if (dx > 0) {
      current_x = (current_x + 1) % cols;
      dx--;
    } else {
      current_x = (current_x - 1 + cols) % cols;
      dx++;
    }
    route.push_back(CoordsToRank(current_x, current_y, rows, cols));
  }

  while (dy != 0) {
    if (dy > 0) {
      current_y = (current_y + 1) % rows;
      dy--;
    } else {
      current_y = (current_y - 1 + rows) % rows;
      dy++;
    }
    route.push_back(CoordsToRank(current_x, current_y, rows, cols));
  }

  return route;
}

void BadanovATorusTopologyMPI::HandleDataRouting(int position_in_route, const std::vector<int> &route,
                                                 const std::vector<double> &data, std::vector<double> &out) {
  if (position_in_route == 0) {
    if (route.size() > 1) {
      int next_hop = route[1];
      MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, next_hop, 0, MPI_COMM_WORLD);
    }
  } else if (position_in_route == static_cast<int>(route.size()) - 1) {
    out.resize(data.size());
    MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    std::vector<double> buffer(data.size());
    MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    int next_hop = route[position_in_route + 1];
    MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, next_hop, 0, MPI_COMM_WORLD);
  }
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

  if (world_size == 1) {
    return true;
  }

  if (world_size == 2) {
    if (world_rank == src_rank) {
      MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, dst_rank, 0, MPI_COMM_WORLD);
    } else if (world_rank == dst_rank) {
      out.resize(data.size());
      MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return true;
  }

  int rows = static_cast<int>(std::sqrt(world_size));
  while (rows > 0 && world_size % rows != 0) {
    rows--;
  }
  int cols = world_size / rows;

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

  HandleDataRouting(position_in_route, route, data, out);

  return true;
}

bool BadanovATorusTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_torus_topology
