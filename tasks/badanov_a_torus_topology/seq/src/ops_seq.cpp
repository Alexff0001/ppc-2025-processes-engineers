#include "badanov_a_torus_topology/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "badanov_a_torus_topology/common/include/common.hpp"
#include "util/include/util.hpp"

namespace badanov_a_torus_topology {

BadanovATorusTopologySEQ::BadanovATorusTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovATorusTopologySEQ::ValidationImpl() {
  const auto& input = GetInput();
  return !std::get<2>(in).empty();
}

bool BadanovATorusTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

int IvanovAMeshTorusSEQ::GetNeighbor(int rank, int direction, int rows, int cols) {
  int row = rank / cols;
  int col = rank % cols;

  switch (direction) {
    case 0: // вверх
      row = (row - 1 + rows) % rows;
      break;
      
    case 1: // вниз
      row = (row + 1) % rows;
      break;
      
    case 2: // влево
      col = (col - 1 + cols) % cols;
      break;
      
    case 3: // вправо
      col = (col + 1) % cols;
      break;
  }
  return row * cols + col;
}

std::vector<int> IvanovAMeshTorusSEQ::FindRoute(int src, int dst, int rows, int cols) {
  std::vector<int> route;

  if (src == dst) {
    route.push_back(src);
    return route;
  }

  int current = src;
  route.push_back(current);

  int current_row = current / cols;
  int current_col = current % cols;

  int target_row = dst / cols;
  int target_col = dst % cols;

  while (current_row != target_row) {
    int direction = (target_row > current_row) ? 1 : 0;
    current = GetNeighbor(current, direction, rows, cols);
    route.push_back(current);
    current_row = current / cols;
  }

  while (current_col != target_col) {
    int direction = (target_col > current_col) ? 3 : 2;
    current = GetNeighbor(current, direction, rows, cols);
    route.push_back(current);
    current_col = current % cols;
  }
  return route;
}

bool BadanovATorusTopologySEQ::RunImpl() {
  const auto &in = GetInput();
  int src = std::get<0>(in);
  int dst = std::get<1>(in);
  
  const std::vector<int> &data = std::get<2>(in);
  constexpr int network_size = 64;

  constexpr int rows = 8;
  constexpr int cols = 8;

  int adjusted_src = src % network_size;
  int adjusted_dst = dst % network_size;

  std::vector<int> route = FindRoute(adjusted_src, adjusted_dst, rows, cols);
  if (route.empty()) {
    GetOutput() = data;
  } else {
    std::vector<int> result;
    result.push_back(static_cast<int>(route.size()));
    result.insert(result.end(), route.begin(), route.end());

    if (!data.empty()) {
      int data_sum = 0;
      for (int value : data) {
        data_sum += value;
      }
      result.push_back(data_sum);
      result.push_back(src);
      result.push_back(dst);
    }

    GetOutput() = result;
  }
  
  return true;
}

bool BadanovATorusTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_torus_topology
