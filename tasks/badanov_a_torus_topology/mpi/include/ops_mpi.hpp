#pragma once

#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BadanovATorusTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static TorusCoords RankToCoords(int rank, int rows, int cols);
  static int CoordsToRank(int x, int y, int rows, int cols);
  static std::vector<int> GetRoute(int src_rank, int dst_rank, int rows, int cols);
  static void HandleDataRouting(int position_in_route, const std::vector<int> &route, const std::vector<double> &data,
                                std::vector<double> &out);
};

}  // namespace badanov_a_torus_topology
