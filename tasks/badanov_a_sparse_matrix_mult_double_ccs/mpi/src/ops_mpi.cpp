#include "badanov_a_sparse_matrix_mult_double_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

BadanovASparseMatrixMultDoubleCcsMPI::BadanovASparseMatrixMultDoubleCcsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovASparseMatrixMultDoubleCcsMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &values_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  if (rows_a <= 0 || cols_a <= 0 || cols_b <= 0) {
    return false;
  }
  if (values_a.size() != row_indices_a.size()) {
    return false;
  }
  if (col_pointers_a.size() != static_cast<size_t>(cols_a) + 1) {
    return false;
  }
  if (values_b.size() != row_indices_b.size()) {
    return false;
  }
  if (col_pointers_b.size() != static_cast<size_t>(cols_b) + 1) {
    return false;
  }

  if (rows_a % world_size != 0 && world_size > rows_a) {
    return false;
  }

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsMPI::PreProcessingImpl() {
  return true;
}

BadanovASparseMatrixMultDoubleCcsMPI::LocalData BadanovASparseMatrixMultDoubleCcsMPI::DistributeDataHorizontal(
    int world_rank, int world_size, const SparseMatrix &a, const SparseMatrix &b) {
  LocalData local;
  local.global_rows = a.rows;
  local.global_inner_dim = a.cols;
  local.global_cols = b.cols;

  int rows_per_proc = a.rows / world_size;
  int local_start_row = world_rank * rows_per_proc;
  int local_end_row = (world_rank + 1) * rows_per_proc;

  std::vector<double> local_a_values;
  std::vector<int> local_a_col_indices;
  std::vector<int> local_a_col_pointers(a.cols + 1, 0);

  for (int col = 0; col < a.cols; ++col) {
    for (int idx = a.col_pointers[col]; idx < a.col_pointers[col + 1]; ++idx) {
      int row = a.row_indices[idx];
      if (row >= local_start_row && row < local_end_row) {
        local_a_col_pointers[col + 1]++;
      }
    }
  }

  for (int col = 0; col < a.cols; ++col) {
    local_a_col_pointers[col + 1] += local_a_col_pointers[col];
  }

  int local_nnz = local_a_col_pointers[a.cols];
  local_a_values.resize(local_nnz);
  local_a_col_indices.resize(local_nnz);

  std::vector<int> current_pos(a.cols, 0);
  for (int col = 0; col < a.cols; ++col) {
    for (int idx = a.col_pointers[col]; idx < a.col_pointers[col + 1]; ++idx) {
      int row = a.row_indices[idx];
      if (row >= local_start_row && row < local_end_row) {
        int local_row = row - local_start_row;
        int local_idx = local_a_col_pointers[col] + current_pos[col];
        local_a_values[local_idx] = a.values[idx];
        local_a_col_indices[local_idx] = local_row;
        current_pos[col]++;
      }
    }
  }

  local.a_local.values = local_a_values;
  local.a_local.row_indices = local_a_col_indices;
  local.a_local.col_pointers = local_a_col_pointers;
  local.a_local.rows = rows_per_proc;
  local.a_local.cols = a.cols;

  local.b_local = b;
  local.b_local.rows = b.rows;
  local.b_local.cols = b.cols;

  return local;
}

std::vector<double> BadanovASparseMatrixMultDoubleCcsMPI::SparseDotProduct(const SparseMatrix &a, const SparseMatrix &b,
                                                                           int col_b) {
  std::vector<double> result(a.rows, 0.0);

  for (int idx_b = b.col_pointers[col_b]; idx_b < b.col_pointers[col_b + 1]; ++idx_b) {
    int row_b = b.row_indices[idx_b];
    double val_b = b.values[idx_b];

    for (int idx_a = a.col_pointers[row_b]; idx_a < a.col_pointers[row_b + 1]; ++idx_a) {
      int row_a = a.row_indices[idx_a];
      double val_a = a.values[idx_a];
      result[row_a] += val_a * val_b;
    }
  }

  return result;
}

SparseMatrix BadanovASparseMatrixMultDoubleCcsMPI::MultiplyLocal(const LocalData &local) {
  std::vector<double> values_c;
  std::vector<int> row_indices_c;
  std::vector<int> col_pointers_c(local.global_cols + 1, 0);

  for (int col_b = 0; col_b < local.global_cols; ++col_b) {
    std::vector<double> local_col = SparseDotProduct(local.a_local, local.a_local, col_b);

    for (int row = 0; row < local.a_local.rows; ++row) {
      if (std::abs(local_col[row]) > 1e-10) {
        values_c.push_back(local_col[row]);
        row_indices_c.push_back(row);
        col_pointers_c[col_b + 1]++;
      }
    }
  }

  for (int col = 0; col < local.global_cols; ++col) {
    col_pointers_c[col + 1] += col_pointers_c[col];
  }

  SparseMatrix c_local;
  c_local.values = values_c;
  c_local.row_indices = row_indices_c;
  c_local.col_pointers = col_pointers_c;
  c_local.rows = local.a_local.rows;
  c_local.cols = local.global_cols;

  return c_local;
}

void BadanovASparseMatrixMultDoubleCcsMPI::GatherResults(int world_rank, int world_size, const SparseMatrix &local_c,
                                                         SparseMatrix &global_c) {
  std::vector<int> local_nnz_per_col(local_c.cols, 0);
  for (int col = 0; col < local_c.cols; ++col) {
    local_nnz_per_col[col] = local_c.col_pointers[col + 1] - local_c.col_pointers[col];
  }

  std::vector<int> global_nnz_per_col(local_c.cols, 0);
  MPI_Allreduce(local_nnz_per_col.data(), global_nnz_per_col.data(), local_c.cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  global_c.col_pointers.resize(local_c.cols + 1, 0);
  for (int col = 0; col < local_c.cols; ++col) {
    global_c.col_pointers[col + 1] = global_c.col_pointers[col] + global_nnz_per_col[col];
  }

  int total_nnz = global_c.col_pointers[local_c.cols];
  global_c.values.resize(total_nnz);
  global_c.row_indices.resize(total_nnz);
  global_c.rows = local_c.rows * world_size;
  global_c.cols = local_c.cols;

  std::vector<int> displs(world_size, 0);
  std::vector<int> recvcounts(world_size, 0);

  for (int col = 0; col < local_c.cols; ++col) {
    int local_start = local_c.col_pointers[col];
    int local_count = local_c.col_pointers[col + 1] - local_start;

    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
      displs[0] = global_c.col_pointers[col];
      for (int i = 1; i < world_size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
      }
    }

    if (local_count > 0) {
      MPI_Gatherv(&local_c.values[local_start], local_count, MPI_DOUBLE, global_c.values.data(), recvcounts.data(),
                  displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

      std::vector<int> adjusted_rows(local_count);
      int row_offset = world_rank * local_c.rows;
      for (int i = 0; i < local_count; ++i) {
        adjusted_rows[i] = local_c.row_indices[local_start + i] + row_offset;
      }

      MPI_Gatherv(adjusted_rows.data(), local_count, MPI_INT, global_c.row_indices.data(), recvcounts.data(),
                  displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
      MPI_Gatherv(nullptr, 0, MPI_DOUBLE, global_c.values.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

      MPI_Gatherv(nullptr, 0, MPI_INT, global_c.row_indices.data(), recvcounts.data(), displs.data(), MPI_INT, 0,
                  MPI_COMM_WORLD);
    }
  }

  if (world_rank != 0) {
    global_c.values.resize(total_nnz);
    global_c.row_indices.resize(total_nnz);
    global_c.col_pointers.resize(local_c.cols + 1);
  }

  MPI_Bcast(global_c.values.data(), total_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_c.row_indices.data(), total_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_c.col_pointers.data(), local_c.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

bool BadanovASparseMatrixMultDoubleCcsMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &values_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  SparseMatrix a_global;
  a_global.values = values_a;
  a_global.row_indices = row_indices_a;
  a_global.col_pointers = col_pointers_a;
  a_global.rows = rows_a;
  a_global.cols = cols_a;

  SparseMatrix b_global;
  b_global.values = values_b;
  b_global.row_indices = row_indices_b;
  b_global.col_pointers = col_pointers_b;
  b_global.rows = cols_a;
  b_global.cols = cols_b;

  LocalData local_data = DistributeDataHorizontal(world_rank, world_size, a_global, b_global);

  SparseMatrix local_c = MultiplyLocal(local_data);

  SparseMatrix global_c;
  GatherResults(world_rank, world_size, local_c, global_c);

  GetOutput() = std::make_tuple(global_c.values, global_c.row_indices, global_c.col_pointers);

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
