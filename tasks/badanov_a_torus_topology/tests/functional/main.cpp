#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "badanov_a_torus_topology/mpi/include/ops_mpi.hpp"
#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(full_param);

    const size_t msg_size = std::get<0>(params);
    const size_t pattern = std::get<1>(params);

    const bool is_seq = (task_name.find("seq_enabled") != std::string::npos);
    const bool is_mpi = (task_name.find("mpi_enabled") != std::string::npos);
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (is_mpi && mpi_initialized == 0) {
      GTEST_SKIP() << "MPI is not initialized (test is running without mpiexec). Skipping MPI tests.";
    }

    int src = 0;
    int dst = 0;

    if (is_seq) {
      src = 0;
      dst = 0;
    } else {
      int world_size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      switch (pattern % 4) {
        case 0:
          src = 0;
          dst = world_size - 1;  // Противоположный узел
          break;
        case 1:
          src = 1;
          dst = 0;  // Обратно к первому
          break;
        case 2:
          if (world_size >= 3) {
            src = 1;
            dst = 2;  // Соседние узлы
          } else {
            src = 0;
            dst = world_size - 1;
          }
          break;
        case 3:
          src = 0;
          dst = 0;  // Отправка самому себе
          break;
        default:
          src = 0;
          dst = world_size - 1;
          break;
      }
    }

    std::vector<double> data(msg_size);
    for (size_t i = 0; i < msg_size; ++i) {
      data[i] = static_cast<double>(i + pattern);
    }
    input_data_ = std::make_tuple(static_cast<size_t>(src), static_cast<size_t>(dst), std::move(data));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &full_param = GetParam();
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    const bool is_seq = (task_name.find("seq_enabled") != std::string::npos);

    const auto &in = input_data_;
    const int dst = static_cast<int>(std::get<1>(in));
    const auto &data = std::get<2>(in);

    if (is_seq) {
      return output_data == data;
    }

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized == 0) {
      return true;
    }

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == dst) {
      return output_data == data;
    }
    return output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(3, 3),     std::make_tuple(2, 5),   std::make_tuple(10, 70),     std::make_tuple(1, 1),
    std::make_tuple(1, 100),   std::make_tuple(100, 1), std::make_tuple(1000, 1000), std::make_tuple(10, 2),
    std::make_tuple(5, 3),     std::make_tuple(4, 5),   std::make_tuple(4, 3),       std::make_tuple(10000, 3),
    std::make_tuple(3, 10000), std::make_tuple(500, 1), std::make_tuple(1, 500)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BadanovATorusTopologyMPI, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology),
    ppc::util::AddFuncTask<BadanovATorusTopologySEQ, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BadanovATorusTopologyFuncTests::PrintFuncTestName<BadanovATorusTopologyFuncTests>;

TEST_P(BadanovATorusTopologyFuncTests, TorusTopologyRouting) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TorusTopologyTests, BadanovATorusTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace badanov_a_torus_topology
