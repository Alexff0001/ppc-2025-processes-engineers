#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
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
    int world_size = 1;

    if (is_seq) {
      src = 0;
      dst = 0;
    } else {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      int rows = 0;
      int cols = 0;
      switch (pattern % 6) {
        case 0:
          rows = 2;
          cols = (world_size + 1) / 2;  // 2xN
          break;
        case 1:
          rows = 3;
          cols = (world_size + 2) / 3;  // 3xN
          break;
        case 2:
          rows = 4;
          cols = (world_size + 3) / 4;  // 4xN
          break;
        case 3:
          rows = 1;
          cols = world_size;  // 1xN
          break;
        case 4:
          rows = world_size;
          cols = 1;  // Nx1
          break;
        case 5:
          rows = static_cast<int>(std::sqrt(world_size));
          if (rows < 1) {
            rows = 1;
          }
          while (rows > 0 && world_size % rows != 0) {
            rows--;
          }
          if (rows == 0) {
            rows = 1;
          }
          cols = world_size / rows;
          break;
        default:
          src = 0;
          dst = std::max(0, world_size - 1);
          break;
      }

      if (rows * cols != world_size) {
        rows = static_cast<int>(std::sqrt(world_size));
        while (rows > 0 && world_size % rows != 0) {
          rows--;
        }
        cols = world_size / rows;
      }

      switch ((pattern / 6) % 7) {
        case 0:
          src = 0;
          dst = 0;
          break;
        case 1:
          src = 0;
          dst = 1 % world_size;
          break;
        case 2:
          src = 0;
          dst = cols % world_size;
          break;
        case 3:
          src = 0;
          dst = (cols + 1) % world_size;
          break;
        case 4:
          src = 0;
          dst = world_size - 1;
          break;
        case 5:
          src = 0;
          dst = (cols - 1) % world_size;
          break;
        case 6:
          switch (pattern % 4) {
            case 0:
              src = 0;
              dst = std::max(0, world_size - 1);
              break;
            case 1:
              src = std::min(1, world_size - 1);
              dst = 0;
              break;
            case 2:
              if (world_size >= 3) {
                src = 1;
                dst = 2;
              } else {
                src = 0;
                dst = world_size - 1;
              }
              break;
            case 3:
              src = 0;
              dst = 0;
              break;
            default:
              src = 0;
              dst = std::max(0, world_size - 1);
              break;
          }
          break;
        default:
          src = 0;
          dst = world_size - 1;
          break;
      }
      src = std::clamp(src, 0, world_size - 1);
      dst = std::clamp(dst, 0, world_size - 1);
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
      const size_t src = std::get<0>(in);
      const size_t dst = std::get<1>(in);

      if (src == dst) {
        return output_data.size() == data.size() && output_data == data;
      } else {
        if (output_data.empty() || output_data.size() != data.size()) {
          return false;
        }

        const int grid_size = 10;
        const int virtual_size = grid_size * grid_size;
        int src_rank = static_cast<int>(src) % virtual_size;
        int dst_rank = static_cast<int>(dst) % virtual_size;

        TorusCoords src_coords{};
        src_coords.x = src_rank % grid_size;
        src_coords.y = src_rank / grid_size;

        TorusCoords dst_coords{};
        dst_coords.x = dst_rank % grid_size;
        dst_coords.y = dst_rank / grid_size;

        int dx = std::abs(dst_coords.x - src_coords.x);
        int dy = std::abs(dst_coords.y - src_coords.y);

        dx = std::min(dx, grid_size - dx);
        dy = std::min(dy, grid_size - dy);

        double distance = std::sqrt(static_cast<double>((dx * dx) + (dy * dy)));
        double scale = 1.0 / (1.0 + distance);

        const double epsilon = 1e-9;
        for (size_t i = 0; i < data.size(); ++i) {
          double expected = data[i] * scale;
          if (std::abs(output_data[i] - expected) > epsilon) {
            return false;
          }
        }
        return true;
      }
    }

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized == 0) {
      return true;
    }

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == dst) {
      return output_data.size() == data.size() && output_data == data;
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

const std::array<TestType, 28> kTestParam = {
    std::make_tuple(3, 0),    std::make_tuple(3, 3),     std::make_tuple(10, 70),     std::make_tuple(1, 1),
    std::make_tuple(1, 2),    std::make_tuple(1, 100),   std::make_tuple(1000, 1000), std::make_tuple(10, 1),
    std::make_tuple(10, 2),   std::make_tuple(5, 1),     std::make_tuple(5, 2),       std::make_tuple(5, 3),
    std::make_tuple(4, 3),    std::make_tuple(10000, 3), std::make_tuple(3, 10000),   std::make_tuple(1, 500),
    std::make_tuple(500, 0),  std::make_tuple(500, 1),   std::make_tuple(1000, 0),    std::make_tuple(500, 2),
    std::make_tuple(10, 100), std::make_tuple(1, 0),     std::make_tuple(0, 0),       std::make_tuple(50, 10),
    std::make_tuple(100, 5),  std::make_tuple(20, 15),   std::make_tuple(7, 8),       std::make_tuple(15, 20)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BadanovATorusTopologyMPI, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology),
    ppc::util::AddFuncTask<BadanovATorusTopologySEQ, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BadanovATorusTopologyFuncTests::PrintFuncTestName<BadanovATorusTopologyFuncTests>;

TEST_P(BadanovATorusTopologyFuncTests, TorusTopologyRouting) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TorusTopologyFuncTests, BadanovATorusTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace badanov_a_torus_topology
