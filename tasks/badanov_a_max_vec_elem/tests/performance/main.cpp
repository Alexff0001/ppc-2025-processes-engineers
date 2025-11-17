#include <gtest/gtest.h>

#include "badanov_a_max_vec_elem/common/include/common.hpp"
#include "badanov_a_max_vec_elem/mpi/include/ops_mpi.hpp"
#include "badanov_a_max_vec_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_max_vec_elem {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BadanovAMaxVecElemMPI, BadanovAMaxVecElemSEQ>(PPC_SETTINGS_badanov_a_max_vec_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace badanov_a_max_vec_elem
