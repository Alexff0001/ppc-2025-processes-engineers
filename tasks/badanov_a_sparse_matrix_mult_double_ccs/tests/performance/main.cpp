#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/mpi/include/ops_mpi.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &test_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int matrix_size = 200;  // default

    if (test_name.find("small") != std::string::npos) {
      matrix_size = 50;
    } else if (test_name.find("medium") != std::string::npos) {
      matrix_size = 200;
    } else if (test_name.find("large") != std::string::npos) {
      matrix_size = 500;
    } else if (test_name.find("huge") != std::string::npos) {
      matrix_size = 1000;
    }

    int rows = matrix_size;
    int inner_dim = matrix_size;
    int cols = matrix_size;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> value_dist(0.0, 10.0);
    std::bernoulli_distribution sparse_dist(0.01);  // 1% non-zero

    std::vector<double> valuesA;
    std::vector<int> row_indicesA;
    std::vector<int> col_pointersA(inner_dim + 1, 0);

    int nnzA = 0;
    for (int col = 0; col < inner_dim; ++col) {
      col_pointersA[col] = nnzA;
      for (int row = 0; row < rows; ++row) {
        if (sparse_dist(gen)) {
          valuesA.push_back(value_dist(gen));
          row_indicesA.push_back(row);
          nnzA++;
        }
      }
    }
    col_pointersA[inner_dim] = nnzA;

    std::vector<double> valuesB;
    std::vector<int> row_indicesB;
    std::vector<int> col_pointersB(cols + 1, 0);

    int nnzB = 0;
    for (int col = 0; col < cols; ++col) {
      col_pointersB[col] = nnzB;
      for (int row = 0; row < inner_dim; ++row) {
        if (sparse_dist(gen)) {
          valuesB.push_back(value_dist(gen));
          row_indicesB.push_back(row);
          nnzB++;
        }
      }
    }
    col_pointersB[cols] = nnzB;

    test_input_ = std::make_tuple(valuesA, row_indicesA, col_pointersA, valuesB, row_indicesB, col_pointersB, rows,
                                  inner_dim, cols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const auto &in = test_input_;
    int cols = std::get<8>(in);

    const auto &col_pointersC = std::get<2>(output_data);

    if (col_pointersC.size() != static_cast<size_t>(cols + 1)) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(BadanovASparseMatrixMultDoubleCcsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BadanovASparseMatrixMultDoubleCcsMPI, BadanovASparseMatrixMultDoubleCcsSEQ>(
        PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovASparseMatrixMultDoubleCcsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovASparseMatrixMultDoubleCcsPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
