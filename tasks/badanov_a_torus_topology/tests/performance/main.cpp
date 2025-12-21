#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"
#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &test_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t msg_size = 1000000;
    int src = 0;
    int dst = world_size - 1;

    if (test_name.find("small") != std::string::npos) {
      msg_size = 10000;
      src = 0;
      dst = std::min(1, world_size - 1);
    } else if (test_name.find("medium") != std::string::npos) {
      msg_size = 100000;
      src = 0;
      dst = world_size / 2;
    } else if (test_name.find("large") != std::string::npos) {
      msg_size = 1000000;
      src = 0;
      dst = world_size - 1;
    } else if (test_name.find("self") != std::string::npos) {
      msg_size = 500000;
      src = 0;
      dst = 0;
    } else if (test_name.find("neighbor") != std::string::npos) {
      msg_size = 500000;
      src = 0;
      dst = 1;
    }

    std::vector<double> data(msg_size);
    for (size_t i = 0; i < msg_size; ++i) {
      data[i] = static_cast<double>(i);
    }

    test_input_ = std::make_tuple(static_cast<size_t>(src), static_cast<size_t>(dst), std::move(data));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const auto &in = test_input_;
    const int dst = static_cast<int>(std::get<1>(in));
    const auto &data = std::get<2>(in);

    if (world_rank == dst) {
      return output_data == data;
    }

    return output_data.empty();
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(BadanovATorusTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = std::tuple_cat(
    ppc::util::MakeAllPerfTasks<InType, BadanovATorusTopologyMPI>(PPC_SETTINGS_badanov_a_torus_topology),
    ppc::util::MakeAllPerfTasks<InType, BadanovATorusTopologySEQ>(PPC_SETTINGS_badanov_a_torus_topology));

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovATorusTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovATorusTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_torus_topology
