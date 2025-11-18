#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <cstddef>
#include <climits>

#include "badanov_a_max_vec_elem/common/include/common.hpp"
#include "badanov_a_max_vec_elem/mpi/include/ops_mpi.hpp"
#include "badanov_a_max_vec_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_max_vec_elem {

class BadanovAMaxVecElemPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_;

  void SetUp() override {
    input_data_ = generate_test_vector(kCount_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.empty()) {
      return output_data == INT_MIN;
    }

    int expected_max = input_data_[0];
    for (size_t i = 1; i < input_data_.size(); ++i) {
      expected_max = std::max(input_data_[i], expected_max);
    }
    return expected_max == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static std::vector<int> GenerateTestVector(int size) {
    std::vector<int> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = (i * 17 + 13) % 1000;
    }
    if (!vec.empty()) {
      vec[size / 2] = 1500;
    }
    return vec;
  }
};

TEST_P(BadanovAMaxVecElemPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BadanovAMaxVecElemMPI, BadanovAMaxVecElemSEQ>(
    PPC_SETTINGS_badanov_a_max_vec_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovAMaxVecElemPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovAMaxVecElemPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_max_vec_elem
