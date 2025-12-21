#pragma once

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_torus_topology {

class BadanovATorusTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BadanovATorusTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static TorusCoords RankToCoords(int rank, int grid_size);
  static int CoordsToRank(int x, int y, int grid_size);
  static double CalculateTorusDistance(const TorusCoords &src, const TorusCoords &dst, int grid_size);
};

}  // namespace badanov_a_torus_topology
