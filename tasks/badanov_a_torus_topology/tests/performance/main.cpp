#include <gtest/gtest.h>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"
#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(
        ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_size = std::get<0>(params);
    
    int source = 0;
    int dest = 0;
    
    if (test_size == 3) {
      source = 0;
      dest = 3;
      input_data_ = {source, dest, 1, 2, 3};
    } else if (test_size == 5) {
      source = 1;
      dest = 6;
      input_data_ = {source, dest};
      for (int i = 0; i < 10; ++i) {
        input_data_.push_back(i * 10);
      }
    } else if (test_size == 7) {
      source = 0;
      dest = 15;
      input_data_ = {source, dest};
      for (int i = 0; i < 100; ++i) {
        input_data_.push_back(i % 256);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }
    
    if (output_data[0] == -1) {
      return output_data.size() == 1;
    }
    
    return output_data.size() >= 2 && output_data[0] >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(BadanovATorusTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kPerfTestParam = {
    std::make_tuple(3, "small_data"),
    std::make_tuple(5, "medium_data"),
    std::make_tuple(7, "large_data")
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BadanovATorusTopologyMPI, BadanovATorusTopologySEQ>(PPC_SETTINGS_badanov_a_torus_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovATorusTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovATorusTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_torus_topology
