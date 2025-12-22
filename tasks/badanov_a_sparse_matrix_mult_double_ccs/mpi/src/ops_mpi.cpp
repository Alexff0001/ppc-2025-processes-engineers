#include "badanov_a_sparse_matrix_mult_double_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

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

  const auto &valuesA = std::get<0>(in);
  const auto &row_indicesA = std::get<1>(in);
  const auto &col_pointersA = std::get<2>(in);
  const auto &valuesB = std::get<3>(in);
  const auto &row_indicesB = std::get<4>(in);
  const auto &col_pointersB = std::get<5>(in);
  int rowsA = std::get<6>(in);
  int colsA = std::get<7>(in);
  int colsB = std::get<8>(in);

  if (rowsA <= 0 || colsA <= 0 || colsB <= 0) {
    return false;
  }
  if (valuesA.size() != row_indicesA.size()) {
    return false;
  }
  if (col_pointersA.size() != static_cast<size_t>(colsA + 1)) {
    return false;
  }
  if (valuesB.size() != row_indicesB.size()) {
    return false;
  }
  if (col_pointersB.size() != static_cast<size_t>(colsB + 1)) {
    return false;
  }

  if (rowsA % world_size != 0 && world_size > rowsA) {
    return false;
  }

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsMPI::PreProcessingImpl() {
  return true;
}

BadanovASparseMatrixMultDoubleCcsMPI::LocalData BadanovASparseMatrixMultDoubleCcsMPI::distributeDataHorizontal(
    int world_rank, int world_size, const SparseMatrix &A, const SparseMatrix &B) {
  LocalData local;
  local.global_rows = A.rows;
  local.global_inner_dim = A.cols;
  local.global_cols = B.cols;

  int rows_per_proc = A.rows / world_size;
  int local_start_row = world_rank * rows_per_proc;
  int local_end_row = (world_rank + 1) * rows_per_proc;

  std::vector<double> local_A_values;
  std::vector<int> local_A_col_indices;
  std::vector<int> local_A_col_pointers(A.cols + 1, 0);

  for (int col = 0; col < A.cols; ++col) {
    for (int idx = A.col_pointers[col]; idx < A.col_pointers[col + 1]; ++idx) {
      int row = A.row_indices[idx];
      if (row >= local_start_row && row < local_end_row) {
        local_A_col_pointers[col + 1]++;
      }
    }
  }

  for (int col = 0; col < A.cols; ++col) {
    local_A_col_pointers[col + 1] += local_A_col_pointers[col];
  }

  int local_nnz = local_A_col_pointers[A.cols];
  local_A_values.resize(local_nnz);
  local_A_col_indices.resize(local_nnz);

  std::vector<int> current_pos(A.cols, 0);
  for (int col = 0; col < A.cols; ++col) {
    for (int idx = A.col_pointers[col]; idx < A.col_pointers[col + 1]; ++idx) {
      int row = A.row_indices[idx];
      if (row >= local_start_row && row < local_end_row) {
        int local_row = row - local_start_row;
        int local_idx = local_A_col_pointers[col] + current_pos[col];
        local_A_values[local_idx] = A.values[idx];
        local_A_col_indices[local_idx] = local_row;
        current_pos[col]++;
      }
    }
  }

  local.A_local.values = local_A_values;
  local.A_local.row_indices = local_A_col_indices;
  local.A_local.col_pointers = local_A_col_pointers;
  local.A_local.rows = rows_per_proc;
  local.A_local.cols = A.cols;

  local.B_local = B;
  local.B_local.rows = B.rows;
  local.B_local.cols = B.cols;

  return local;
}

std::vector<double> BadanovASparseMatrixMultDoubleCcsMPI::sparseDotProduct(const SparseMatrix &A, const SparseMatrix &B,
                                                                           int colB) {
  std::vector<double> result(A.rows, 0.0);

  for (int idxB = B.col_pointers[colB]; idxB < B.col_pointers[colB + 1]; ++idxB) {
    int rowB = B.row_indices[idxB];
    double valB = B.values[idxB];

    for (int idxA = A.col_pointers[rowB]; idxA < A.col_pointers[rowB + 1]; ++idxA) {
      int rowA = A.row_indices[idxA];
      double valA = A.values[idxA];
      result[rowA] += valA * valB;
    }
  }

  return result;
}

SparseMatrix BadanovASparseMatrixMultDoubleCcsMPI::multiplyLocal(const LocalData &local) {
  std::vector<double> valuesC;
  std::vector<int> row_indicesC;
  std::vector<int> col_pointersC(local.global_cols + 1, 0);

  for (int colB = 0; colB < local.global_cols; ++colB) {
    std::vector<double> local_col = sparseDotProduct(local.A_local, local.B_local, colB);

    for (int row = 0; row < local.A_local.rows; ++row) {
      if (std::abs(local_col[row]) > 1e-10) {
        valuesC.push_back(local_col[row]);
        row_indicesC.push_back(row);
        col_pointersC[colB + 1]++;
      }
    }
  }

  for (int col = 0; col < local.global_cols; ++col) {
    col_pointersC[col + 1] += col_pointersC[col];
  }

  SparseMatrix C_local;
  C_local.values = valuesC;
  C_local.row_indices = row_indicesC;
  C_local.col_pointers = col_pointersC;
  C_local.rows = local.A_local.rows;
  C_local.cols = local.global_cols;

  return C_local;
}

void BadanovASparseMatrixMultDoubleCcsMPI::gatherResults(int world_rank, int world_size, const SparseMatrix &local_C,
                                                         SparseMatrix &global_C) {
  std::vector<int> local_nnz_per_col(local_C.cols, 0);
  for (int col = 0; col < local_C.cols; ++col) {
    local_nnz_per_col[col] = local_C.col_pointers[col + 1] - local_C.col_pointers[col];
  }

  std::vector<int> global_nnz_per_col(local_C.cols, 0);
  MPI_Allreduce(local_nnz_per_col.data(), global_nnz_per_col.data(), local_C.cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  global_C.col_pointers.resize(local_C.cols + 1, 0);
  for (int col = 0; col < local_C.cols; ++col) {
    global_C.col_pointers[col + 1] = global_C.col_pointers[col] + global_nnz_per_col[col];
  }

  int total_nnz = global_C.col_pointers[local_C.cols];
  global_C.values.resize(total_nnz);
  global_C.row_indices.resize(total_nnz);
  global_C.rows = local_C.rows * world_size;
  global_C.cols = local_C.cols;

  std::vector<int> displs(world_size, 0);
  std::vector<int> recvcounts(world_size, 0);

  for (int col = 0; col < local_C.cols; ++col) {
    int local_start = local_C.col_pointers[col];
    int local_count = local_C.col_pointers[col + 1] - local_start;

    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
      displs[0] = global_C.col_pointers[col];
      for (int i = 1; i < world_size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
      }
    }

    if (local_count > 0) {
      MPI_Gatherv(&local_C.values[local_start], local_count, MPI_DOUBLE, global_C.values.data(), recvcounts.data(),
                  displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

      std::vector<int> adjusted_rows(local_count);
      int row_offset = world_rank * local_C.rows;
      for (int i = 0; i < local_count; ++i) {
        adjusted_rows[i] = local_C.row_indices[local_start + i] + row_offset;
      }

      MPI_Gatherv(adjusted_rows.data(), local_count, MPI_INT, global_C.row_indices.data(), recvcounts.data(),
                  displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
      MPI_Gatherv(nullptr, 0, MPI_DOUBLE, global_C.values.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

      MPI_Gatherv(nullptr, 0, MPI_INT, global_C.row_indices.data(), recvcounts.data(), displs.data(), MPI_INT, 0,
                  MPI_COMM_WORLD);
    }
  }

  if (world_rank != 0) {
    global_C.values.resize(total_nnz);
    global_C.row_indices.resize(total_nnz);
    global_C.col_pointers.resize(local_C.cols + 1);
  }

  MPI_Bcast(global_C.values.data(), total_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_C.row_indices.data(), total_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_C.col_pointers.data(), local_C.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

bool BadanovASparseMatrixMultDoubleCcsMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();

  const auto &valuesA = std::get<0>(in);
  const auto &row_indicesA = std::get<1>(in);
  const auto &col_pointersA = std::get<2>(in);
  const auto &valuesB = std::get<3>(in);
  const auto &row_indicesB = std::get<4>(in);
  const auto &col_pointersB = std::get<5>(in);
  int rowsA = std::get<6>(in);
  int colsA = std::get<7>(in);
  int colsB = std::get<8>(in);

  SparseMatrix A_global;
  A_global.values = valuesA;
  A_global.row_indices = row_indicesA;
  A_global.col_pointers = col_pointersA;
  A_global.rows = rowsA;
  A_global.cols = colsA;

  SparseMatrix B_global;
  B_global.values = valuesB;
  B_global.row_indices = row_indicesB;
  B_global.col_pointers = col_pointersB;
  B_global.rows = colsA;
  B_global.cols = colsB;

  LocalData local_data = distributeDataHorizontal(world_rank, world_size, A_global, B_global);

  SparseMatrix local_C = multiplyLocal(local_data);

  SparseMatrix global_C;
  gatherResults(world_rank, world_size, local_C, global_C);

  GetOutput() = std::make_tuple(global_C.values, global_C.row_indices, global_C.col_pointers);

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
