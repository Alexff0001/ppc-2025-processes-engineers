#include "badanov_a_sparse_matrix_mult_double_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

BadanovASparseMatrixMultDoubleCcsSEQ::BadanovASparseMatrixMultDoubleCcsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::ValidationImpl() {
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

  if (rowsA <= 0 || colsA <= 0) {
    return false;
  }
  if (valuesA.size() != row_indicesA.size()) {
    return false;
  }
  if (col_pointersA.size() != static_cast<size_t>(colsA + 1)) {
    return false;
  }

  if (colsB <= 0) {
    return false;
  }
  if (valuesB.size() != row_indicesB.size()) {
    return false;
  }
  if (col_pointersB.size() != static_cast<size_t>(colsB + 1)) {
    return false;
  }

  for (size_t i = 0; i < row_indicesB.size(); ++i) {
    if (row_indicesB[i] < 0 || row_indicesB[i] >= colsA) {
      return false;
    }
  }

  for (size_t i = 0; i < row_indicesA.size(); ++i) {
    if (row_indicesA[i] < 0 || row_indicesA[i] >= rowsA) {
      return false;
    }
  }

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

double BadanovASparseMatrixMultDoubleCcsSEQ::dotProduct(const std::vector<double> &colA,
                                                        const std::vector<double> &colB) {
  double result = 0.0;
  for (size_t i = 0; i < colA.size(); ++i) {
    result += colA[i] * colB[i];
  }
  return result;
}

SparseMatrix BadanovASparseMatrixMultDoubleCcsSEQ::multiplyCCS(const SparseMatrix &A, const SparseMatrix &B) {
  std::vector<double> valuesC;
  std::vector<int> row_indicesC;
  std::vector<int> col_pointersC(B.cols + 1, 0);

  std::vector<std::vector<double>> columnsA(A.cols);
  for (int j = 0; j < A.cols; ++j) {
    columnsA[j] = A.getColumn(j);
  }

  for (int j = 0; j < B.cols; ++j) {
    std::vector<double> colB = B.getColumn(j);

    for (int i = 0; i < A.rows; ++i) {
      double sum = 0.0;

      for (int k = 0; k < A.cols; ++k) {
        sum += columnsA[k][i] * colB[k];
      }

      if (std::abs(sum) > 1e-10) {
        valuesC.push_back(sum);
        row_indicesC.push_back(i);
        col_pointersC[j + 1]++;
      }
    }
  }

  for (int j = 0; j < B.cols; ++j) {
    col_pointersC[j + 1] += col_pointersC[j];
  }

  SparseMatrix C;
  C.values = valuesC;
  C.row_indices = row_indicesC;
  C.col_pointers = col_pointersC;
  C.rows = A.rows;
  C.cols = B.cols;

  return C;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::RunImpl() {
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

  SparseMatrix A;
  A.values = valuesA;
  A.row_indices = row_indicesA;
  A.col_pointers = col_pointersA;
  A.rows = rowsA;
  A.cols = colsA;

  SparseMatrix B;
  B.values = valuesB;
  B.row_indices = row_indicesB;
  B.col_pointers = col_pointersB;
  B.rows = colsA;
  B.cols = colsB;

  SparseMatrix C = multiplyCCS(A, B);

  GetOutput() = std::make_tuple(C.values, C.row_indices, C.col_pointers);

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
