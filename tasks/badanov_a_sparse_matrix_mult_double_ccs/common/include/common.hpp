#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

using InType = std::tuple<std::vector<double>,  // values of matrix A
                          std::vector<int>,     // row indices of matrix A
                          std::vector<int>,     // column pointers of matrix A
                          std::vector<double>,  // values of matrix B
                          std::vector<int>,     // row indices of matrix B
                          std::vector<int>,     // column pointers of matrix B
                          int,                  // rows of A
                          int,                  // cols of A (rows of B)
                          int                   // cols of B
                          >;
using OutType = std::tuple<std::vector<double>,  // values of result matrix C
                           std::vector<int>,     // row indices of result matrix C
                           std::vector<int>      // column pointers of result matrix C
                           >;
using TestType = std::tuple<size_t, size_t, size_t>;  // rows, inner_dim, cols
using BaseTask = ppc::task::Task<InType, OutType>;

struct SparseMatrix {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;
  int rows;
  int cols;

  int nnz() const {
    return values.size();
  }

  std::vector<double> getColumn(int j) const {
    std::vector<double> col(rows, 0.0);
    for (int idx = col_pointers[j]; idx < col_pointers[j + 1]; ++idx) {
      col[row_indices[idx]] = values[idx];
    }
    return col;
  }
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
