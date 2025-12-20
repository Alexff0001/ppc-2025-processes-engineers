#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
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
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);
    
    std::mt19937 rng(test_id);
    std::uniform_int_distribution<int> dist(1, 100);
    
    switch (test_id) {
      case 1:
        input_data_ = {0, 3, 10, 20, 30};
        break;
        
      case 2: 
        input_data_ = {5, 1, 42};
        break;
        
      case 3:
        input_data_ = {2, 2, 1, 2, 3, 4, 5};
        break;
        
      case 4:
        input_data_.push_back(dist(rng) % 8);
        input_data_.push_back(dist(rng) % 8);
        
        int data_size = dist(rng) % 10 + 1;
        for (int i = 0; i < data_size; ++i) {
          input_data_.push_back(dist(rng));
        }
        break;
    
  }

  bool CheckTestOutputData(OutType &output_data) final {
  if (output_data.empty()) {
      return false;
    }
    
    if (output_data[0] == -1) {
      return output_data.size() == 1;
    }

    if (output_data.size() < 2) {
      return false;
    }
    
    int hops_count = output_data[0];
    
    if (hops_count < 0 || hops_count > 100) {
      return false;
    }
    
    if (static_cast<int>(output_data.size()) < 1 + hops_count + 1) {
      return false;
    }
    
    const auto& input = GetTestInputData();
    if (input.size() >= 2) {
      int src = input[0];
      int dst = input[1];
      
      if (output_data[1] != src) {
        return false;
      }
      
      if (output_data[1 + hops_count] != dst) {
        return false;
      }
    }
    
    if (hops_count > 0) {
      std::vector<int> path(output_data.begin() + 1, output_data.begin() + 1 + hops_count + 1);
      std::sort(path.begin(), path.end());
      
      if (std::unique(path.begin(), path.end()) != path.end()) {
        return false;
      }
    }
    
    int result_index = 1 + hops_count + 1;
    if (static_cast<int>(output_data.size()) > result_index) {
      int final_result = output_data[result_index];
      if (final_result < 0) {
        return false;
      }
    }
    
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(BadanovATorusTopologyFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BadanovATorusTopologyMPI, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology),
                   ppc::util::AddFuncTask<BadanovATorusTopologySEQ, InType>(kTestParam, PPC_SETTINGS_badanov_a_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BadanovATorusTopologyFuncTests::PrintFuncTestName<BadanovATorusTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BadanovATorusTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace badanov_a_torus_topology
