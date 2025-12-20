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

std::vector<int> BadanovATorusTopologySEQ::simulateTorusPath(int src, int dst, int total_nodes) {
  std::vector<int> path;
  
  if (src == dst) {
    path.push_back(src);
    return path;
  }
  
  // Предполагаем квадратную сетку
  int grid_size = static_cast<int>(std::sqrt(total_nodes));
  if (grid_size * grid_size != total_nodes) {
    grid_size = total_nodes; // fallback
  }
  
  int src_row = src / grid_size;
  int src_col = src % grid_size;
  int dst_row = dst / grid_size;
  int dst_col = dst % grid_size;
  
  int current = src;
  path.push_back(current);
  
  while (src_row != dst_row) {
    int row_diff = dst_row - src_row;
    
    if (std::abs(row_diff) <= grid_size / 2) {
      src_row += (row_diff > 0) ? 1 : -1;
    } else {
      src_row -= (row_diff > 0) ? 1 : -1;
    }
    
    src_row = (src_row + grid_size) % grid_size;
    current = src_row * grid_size + src_col;
    path.push_back(current);
  }
  
  while (src_col != dst_col) {
    int col_diff = dst_col - src_col;
    
    if (std::abs(col_diff) <= grid_size / 2) {
      src_col += (col_diff > 0) ? 1 : -1;
    } else {
      src_col -= (col_diff > 0) ? 1 : -1;
    }
    
    src_col = (src_col + grid_size) % grid_size;
    current = src_row * grid_size + src_col;
    path.push_back(current);
  }
  
  return path;
}

void BadanovATorusTopologySEQ::printPath(const std::vector<int>& path) {
  std::cout << "SEQ Torus path simulation:" << std::endl;
  std::cout << "Path length: " << path.size() << " hops" << std::endl;
  std::cout << "Path: ";
  for (size_t i = 0; i < path.size(); ++i) {
    std::cout << path[i];
    if (i < path.size() - 1) std::cout << " -> ";
  }
  std::cout << std::endl;
}

bool BadanovATorusTopologySEQ::RunImpl() {
  const auto& input = GetInput();
  int src = input[0];
  int dst = input[1];
  
  int total_nodes = 16;
  
  std::vector<int> path = simulateTorusPath(src, dst, total_nodes);
  
  printPath(path);
  
  GetOutput().push_back(static_cast<int>(path.size()));
  GetOutput().insert(GetOutput().end(), path.begin(), path.end());
  
  if (!path.empty()) {
    GetOutput().push_back(100);
    GetOutput().push_back(src);
    GetOutput().push_back(dst);
  }
  
  return true;
}

bool BadanovATorusTopologySEQ::PostProcessingImpl() {
  if (!GetOutput().empty()) {
    for (size_t i = 0; i < GetOutput().size(); ++i) {
      GetOutput()[i] *= 2;
    }
  }
  return true;
}

}  // namespace badanov_a_torus_topology
