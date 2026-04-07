#pragma once

#include <array>
#include <cstdint>
#include <vector>

struct Point2 {
  double x;
  double y;
};

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

std::vector<Point2> extract_obstacle_boundary(const std::vector<uint8_t>& mask, int w, int h);

void build_safe_halfspaces(
    const Mat2& P,
    const Point2& c,
    const std::vector<Point2>& obs_points,
    int patch_size,
    double safety_margin,
    std::vector<Vec2>& A,
    std::vector<double>& b);

std::vector<Point2> halfspace_intersection_2d(
    const std::vector<Vec2>& A,
    const std::vector<double>& b,
    const Point2& interior_point,
    double eps_det = 1e-10,
    double eps_feas = 1e-6);

double polygon_mask_area(const std::vector<Vec2>& A, const std::vector<double>& b, int patch_size);
double collision_mask_area(const std::vector<Vec2>& A, const std::vector<double>& b, const std::vector<uint8_t>& obs_mask, int patch_size);
