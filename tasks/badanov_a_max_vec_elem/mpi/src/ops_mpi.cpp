#include "badanov_a_max_vec_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "badanov_a_max_vec_elem/common/include/common.hpp"

namespace badanov_a_max_vec_elem {

BadanovAMaxVecElemMPI::BadanovAMaxVecElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BadanovAMaxVecElemMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool BadanovAMaxVecElemMPI::PreProcessingImpl() {
  return true;
}

bool BadanovAMaxVecElemMPI::RunImpl() {
  std::vector<int> &tmp_vec = GetInput();

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_elem = 0;
  if (rank == 0) {
    total_elem = static_cast<int>(tmp_vec.size());
  }

  MPI_Bcast(&total_elem, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elem == 0) {
    return true;
  }

  int base_size = total_elem / world_size;
  int remainder = total_elem % world_size;

  int start_i = 0;
  int end_i = 0;

  if (rank < remainder) {
    start_i = rank * (base_size + 1);
    end_i = start_i + (base_size + 1);
  } else {
    start_i = (remainder * (base_size + 1)) + ((rank - remainder) * base_size);
    end_i = std::min(start_i + base_size, total_elem);
  }

  int max_elem_local = INT_MIN;
  for (int i = start_i; i < end_i; i++) {
    max_elem_local = std::max(tmp_vec[i], max_elem_local);
  }

  int max_elem_global = INT_MIN;

  MPI_Allreduce(&max_elem_local, &max_elem_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = max_elem_global;
  return true;
}

bool BadanovAMaxVecElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_max_vec_elem
