#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <random>

#include "badanov_a_max_vec_elem/common/include/common.hpp"
#include "badanov_a_max_vec_elem/mpi/include/ops_mpi.hpp"
#include "badanov_a_max_vec_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_max_vec_elem {

class BadanovAMaxVecElemPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kCount_ = 1000000;
  InType input_data_;
  OutType expected_val_{};

  void SetUp() override {
    std::random_device rand_dev;
    std::mt19937 gen(rand_dev());
    std::uniform_int_distribution<int> rand(-777, 777);

    input_data_.resize(kCount_);
    for (size_t i = 0; i < kCount_; i++) {
      input_data_[i] = rand(gen);
    }

    expected_val_ = 1337;
    input_data_[kCount_ / 2] = expected_val_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_val_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
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
