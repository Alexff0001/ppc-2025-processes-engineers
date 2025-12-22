#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

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
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "x" +
           std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(full_param);

    const size_t rows = std::get<0>(params);
    const size_t inner_dim = std::get<1>(params);
    const size_t cols = std::get<2>(params);

    const bool is_mpi = (task_name.find("mpi_enabled") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (is_mpi && mpi_initialized == 0) {
      GTEST_SKIP() << "MPI is not initialized (test is running without mpiexec). Skipping MPI tests.";
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> value_dist(0.0, 10.0);
    std::bernoulli_distribution sparse_dist(0.05);

    std::vector<double> valuesA;
    std::vector<int> row_indicesA;
    std::vector<int> col_pointersA(inner_dim + 1, 0);

    int nnzA = 0;
    for (size_t col = 0; col < inner_dim; ++col) {
      col_pointersA[col] = nnzA;
      for (size_t row = 0; row < rows; ++row) {
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
    for (size_t col = 0; col < cols; ++col) {
      col_pointersB[col] = nnzB;
      for (size_t row = 0; row < inner_dim; ++row) {
        if (sparse_dist(gen)) {
          valuesB.push_back(value_dist(gen));
          row_indicesB.push_back(row);
          nnzB++;
        }
      }
    }
    col_pointersB[cols] = nnzB;

    input_data_ = std::make_tuple(valuesA, row_indicesA, col_pointersA, valuesB, row_indicesB, col_pointersB, rows,
                                  inner_dim, cols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &in = input_data_;
    int rows = std::get<6>(in);
    int cols = std::get<8>(in);

    const auto &valuesC = std::get<0>(output_data);
    const auto &row_indicesC = std::get<1>(output_data);
    const auto &col_pointersC = std::get<2>(output_data);

    if (col_pointersC.size() != static_cast<size_t>(cols + 1)) {
      return false;
    }
    if (valuesC.size() != row_indicesC.size()) {
      return false;
    }

    for (size_t i = 0; i < row_indicesC.size(); ++i) {
      if (row_indicesC[i] < 0 || row_indicesC[i] >= rows) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(10, 10, 10),    std::make_tuple(50, 50, 50),      std::make_tuple(100, 100, 100),
    std::make_tuple(200, 200, 200), std::make_tuple(100, 50, 200),  // rectangular
    std::make_tuple(200, 100, 50),                                  // rectangular
    std::make_tuple(500, 500, 500), std::make_tuple(1000, 100, 1000), std::make_tuple(100, 1000, 100),
    std::make_tuple(300, 300, 300), std::make_tuple(400, 200, 400),   std::make_tuple(200, 400, 200),
    std::make_tuple(600, 600, 100), std::make_tuple(100, 600, 600),   std::make_tuple(800, 800, 800)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BadanovASparseMatrixMultDoubleCcsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs),
                                           ppc::util::AddFuncTask<BadanovASparseMatrixMultDoubleCcsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BadanovASparseMatrixMultDoubleCcsFuncTests::PrintFuncTestName<BadanovASparseMatrixMultDoubleCcsFuncTests>;

TEST_P(BadanovASparseMatrixMultDoubleCcsFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BadanovASparseMatrixMultDoubleCcsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
