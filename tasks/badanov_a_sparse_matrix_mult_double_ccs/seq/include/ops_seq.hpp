#pragma once

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BadanovASparseMatrixMultDoubleCcsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static SparseMatrix multiplyCCS(const SparseMatrix &A, const SparseMatrix &B);
  static double dotProduct(const std::vector<double> &colA, const std::vector<double> &colB);
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
