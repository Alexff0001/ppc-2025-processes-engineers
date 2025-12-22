#pragma once

#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BadanovASparseMatrixMultDoubleCcsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct LocalData {
    SparseMatrix A_local;
    SparseMatrix B_local;
    int global_rows;
    int global_inner_dim;
    int global_cols;
  };

  LocalData distributeDataHorizontal(int world_rank, int world_size, const SparseMatrix &A, const SparseMatrix &B);
  SparseMatrix multiplyLocal(const LocalData &local);
  void gatherResults(int world_rank, int world_size, const SparseMatrix &local_C, SparseMatrix &global_C);
  static std::vector<double> sparseDotProduct(const SparseMatrix &A, const SparseMatrix &B, int colB);
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
