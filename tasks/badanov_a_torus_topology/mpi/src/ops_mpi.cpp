#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "util/include/util.hpp"

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
  int src = std::get<0>(in);
  int dst = std::get<1>(in);
  return src >= 0 && dst >= 0 && src < world_size && dst < world_size;
}

bool BadanovATorusTopologyMPI::PreProcessingImpl() {
  return true;
}

int BadanovATorusTopologyMPI::CalculateRows(int world_size) {
  int rows = static_cast<int>(std::sqrt(world_size));
  while (world_size % rows != 0 && rows > 1) {
    --rows;
  }
  return rows;
}

int BadanovATorusTopologyMPI::GetNeighbor(int rank, int direction, int rows, int cols) {
  int row = rank / cols;
  int col = rank % cols;

  switch (direction) {
    case 0:
      row = (row - 1 + rows) % rows;
      break;
    case 1:
      row = (row + 1) % rows;
      break;
    case 2:
      col = (col - 1 + cols) % cols;
      break;
    case 3:
      col = (col + 1) % cols;
      break;
  }

  return row * cols + col;
}

bool BadanovATorusTopologyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  int src = std::get<0>(in);
  int dst = std::get<1>(in);
  const std::vector<int> &data = std::get<2>(in);

  if (size == 1) {
    if (rank == src) {
      GetOutput() = data;
    }
    return true;
  }

  int rows = CalculateRows(size);
  int cols = size / rows;

  if (rank == src) {
    std::vector<int> buffer = data;
    int current_rank = src;
    int current_row = src / cols;
    int current_col = src % cols;
    int target_row = dst / cols;
    int target_col = dst % cols;

    while (current_row != target_row) {
      int direction = (target_row > current_row) ? 1 : 0;
      int next_rank = GetNeighbor(current_rank, direction, rows, cols);
      int count = static_cast<int>(buffer.size());

      MPI_Send(&count, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
      MPI_Send(buffer.data(), count, MPI_INT, next_rank, 1, MPI_COMM_WORLD);

      current_rank = next_rank;
      current_row = current_rank / cols;
    }

    while (current_col != target_col) {
      int direction = (target_col > current_col) ? 3 : 2;
      int next_rank = GetNeighbor(current_rank, direction, rows, cols);
      int count = static_cast<int>(buffer.size());

      MPI_Send(&count, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
      MPI_Send(buffer.data(), count, MPI_INT, next_rank, 1, MPI_COMM_WORLD);

      current_rank = next_rank;
      current_col = current_rank % cols;
    }

    if (src == dst) {
      GetOutput() = buffer;
    }
  }

  if (rank != src) {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      int count = 0;
      MPI_Recv(&count, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> buffer(count);
      MPI_Recv(buffer.data(), count, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (rank == dst) {
        GetOutput() = buffer;
      } else {
        int current_row = rank / cols;
        int current_col = rank % cols;
        int target_row = dst / cols;
        int target_col = dst % cols;

        if (current_row != target_row) {
          int direction = (target_row > current_row) ? 1 : 0;
          int next_rank = GetNeighbor(rank, direction, rows, cols);
          MPI_Send(&count, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
          MPI_Send(buffer.data(), count, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
        } else if (current_col != target_col) {
          int direction = (target_col > current_col) ? 3 : 2;
          int next_rank = GetNeighbor(rank, direction, rows, cols);
          MPI_Send(&count, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
          MPI_Send(buffer.data(), count, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BadanovATorusTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_torus_topology
