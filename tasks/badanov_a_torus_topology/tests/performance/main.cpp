#include <gtest/gtest.h>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"
#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(BadanovATorusTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BadanovATorusTopologyMPI, BadanovATorusTopologySEQ>(PPC_SETTINGS_example_processes_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovATorusTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovATorusTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_torus_topology
