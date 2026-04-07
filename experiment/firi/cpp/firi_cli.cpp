#include <gcopter/firi.hpp>

#include <Eigen/Eigen>

#include <iostream>
#include <vector>

int main() {
  double cx = 64.0;
  double cy = 64.0;
  int patch_size = 128;
  int n = 0;

  if (!(std::cin >> cx >> cy)) {
    std::cout << "ERR invalid_center" << std::endl;
    return 1;
  }
  if (!(std::cin >> patch_size)) {
    std::cout << "ERR invalid_patch_size" << std::endl;
    return 1;
  }
  if (!(std::cin >> n) || n < 0) {
    std::cout << "ERR invalid_obs_count" << std::endl;
    return 1;
  }

  Eigen::Matrix3Xd pc(3, std::max(1, n));
  for (int i = 0; i < n; ++i) {
    double x = 0.0, y = 0.0;
    if (!(std::cin >> x >> y)) {
      std::cout << "ERR invalid_obs_point" << std::endl;
      return 1;
    }
    pc(0, i) = x;
    pc(1, i) = y;
    pc(2, i) = 0.0;
  }
  if (n == 0) {
    pc(0, 0) = -1.0e6;
    pc(1, 0) = -1.0e6;
    pc(2, 0) = 0.0;
  }

  // Domain halfspaces: n.dot(x) + d <= 0
  Eigen::MatrixX4d bd(6, 4);
  bd.setZero();
  const double s = static_cast<double>(patch_size - 1);

  bd(0, 0) = 1.0;   bd(0, 3) = -s;   // x <= s
  bd(1, 0) = -1.0;  bd(1, 3) = 0.0;  // x >= 0
  bd(2, 1) = 1.0;   bd(2, 3) = -s;   // y <= s
  bd(3, 1) = -1.0;  bd(3, 3) = 0.0;  // y >= 0
  bd(4, 2) = 1.0;   bd(4, 3) = -1.0; // z <= 1
  bd(5, 2) = -1.0;  bd(5, 3) = -1.0; // z >= -1

  Eigen::Vector3d a(cx, cy, 0.0);
  Eigen::Vector3d b(cx, cy, 0.0);

  Eigen::MatrixX4d hp;
  const bool ok = firi::firi(bd, pc.leftCols(std::max(1, n)), a, b, hp, 5);
  if (!ok || hp.rows() <= 0) {
    std::cout << "ERR no_hpoly" << std::endl;
    return 2;
  }

  // Keep only constraints that affect xy-plane at z=0:
  // nx*x + ny*y + d <= 0
  std::vector<Eigen::Vector3d> rows;
  rows.reserve(hp.rows());
  for (int i = 0; i < hp.rows(); ++i) {
    const double nx = hp(i, 0);
    const double ny = hp(i, 1);
    const double d = hp(i, 3);
    const double norm = std::sqrt(nx * nx + ny * ny);
    if (norm < 1.0e-8) {
      continue;
    }
    rows.emplace_back(nx / norm, ny / norm, d / norm);
  }

  if (rows.size() < 3) {
    std::cout << "ERR insufficient_2d_constraints" << std::endl;
    return 3;
  }

  std::cout << "OK " << rows.size() << std::endl;
  for (const auto &r : rows) {
    std::cout << r(0) << " " << r(1) << " " << r(2) << std::endl;
  }

  return 0;
}
