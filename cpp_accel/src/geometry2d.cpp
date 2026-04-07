#include "geometry2d.hpp"

#include <algorithm>
#include <cmath>

namespace {

double dot(const Vec2& a, const Point2& p) {
  return a[0] * p.x + a[1] * p.y;
}

Point2 solve_line_intersection(const Vec2& a1, double c1, const Vec2& a2, double c2, bool& ok, double eps_det) {
  const double det = a1[0] * a2[1] - a1[1] * a2[0];
  if (std::abs(det) <= eps_det) {
    ok = false;
    return {0.0, 0.0};
  }
  ok = true;
  const double x = (c1 * a2[1] - a1[1] * c2) / det;
  const double y = (a1[0] * c2 - c1 * a2[0]) / det;
  return {x, y};
}

bool feasible(const std::vector<Vec2>& A, const std::vector<double>& b, const Point2& p, double eps_feas) {
  for (size_t i = 0; i < A.size(); ++i) {
    if (dot(A[i], p) > b[i] + eps_feas) {
      return false;
    }
  }
  return true;
}

bool almost_same(const Point2& p, const Point2& q, double tol2 = 1e-12) {
  const double dx = p.x - q.x;
  const double dy = p.y - q.y;
  return (dx * dx + dy * dy) <= tol2;
}

}  // namespace

std::vector<Point2> extract_obstacle_boundary(const std::vector<uint8_t>& mask, int w, int h) {
  std::vector<Point2> out;
  out.reserve(static_cast<size_t>(w * h / 4));

  auto at = [&](int x, int y) -> bool {
    if (x < 0 || x >= w || y < 0 || y >= h) {
      return false;
    }
    return mask[static_cast<size_t>(y * w + x)] != 0;
  };

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (!at(x, y)) {
        continue;
      }
      const bool up = at(x, y - 1);
      const bool down = at(x, y + 1);
      const bool left = at(x - 1, y);
      const bool right = at(x + 1, y);
      const bool interior = up && down && left && right;
      if (!interior) {
        out.push_back({static_cast<double>(x), static_cast<double>(y)});
      }
    }
  }

  return out;
}

void build_safe_halfspaces(
    const Mat2& P,
    const Point2& c,
    const std::vector<Point2>& obs_points,
    int patch_size,
    double safety_margin,
    std::vector<Vec2>& A,
    std::vector<double>& b) {
  A.clear();
  b.clear();

  A.push_back({1.0, 0.0});
  b.push_back(static_cast<double>(patch_size) - safety_margin);
  A.push_back({-1.0, 0.0});
  b.push_back(safety_margin);
  A.push_back({0.0, 1.0});
  b.push_back(static_cast<double>(patch_size) - safety_margin);
  A.push_back({0.0, -1.0});
  b.push_back(safety_margin);

  if (obs_points.empty()) {
    return;
  }

  struct ObsDist {
    Point2 p;
    double d;
  };
  std::vector<ObsDist> sorted;
  sorted.reserve(obs_points.size());

  for (const auto& p : obs_points) {
    const double dx = p.x - c.x;
    const double dy = p.y - c.y;
    const double qx = P[0][0] * dx + P[0][1] * dy;
    const double qy = P[1][0] * dx + P[1][1] * dy;
    const double d = dx * qx + dy * qy;
    sorted.push_back({p, d});
  }

  std::sort(sorted.begin(), sorted.end(), [](const ObsDist& a, const ObsDist& b) {
    return a.d < b.d;
  });

  std::vector<Point2> active;
  active.reserve(sorted.size());
  for (const auto& od : sorted) {
    active.push_back(od.p);
  }

  while (!active.empty()) {
    const Point2 obs = active.front();
    const double vx = obs.x - c.x;
    const double vy = obs.y - c.y;

    const double nx = P[0][0] * vx + P[0][1] * vy;
    const double ny = P[1][0] * vx + P[1][1] * vy;
    const double n_norm = std::sqrt(nx * nx + ny * ny);

    std::vector<Point2> next_active;
    next_active.reserve(active.size());

    if (n_norm > 1e-6) {
      const Vec2 n_hat = {nx / n_norm, ny / n_norm};
      const double b_val = n_hat[0] * obs.x + n_hat[1] * obs.y - safety_margin;
      A.push_back(n_hat);
      b.push_back(b_val);

      for (size_t i = 1; i < active.size(); ++i) {
        const Point2& p = active[i];
        const double proj = n_hat[0] * p.x + n_hat[1] * p.y;
        if (proj <= b_val + 1e-3) {
          next_active.push_back(p);
        }
      }
    } else {
      for (size_t i = 1; i < active.size(); ++i) {
        next_active.push_back(active[i]);
      }
    }

    active.swap(next_active);
  }
}

std::vector<Point2> halfspace_intersection_2d(
    const std::vector<Vec2>& A,
    const std::vector<double>& b,
    const Point2& interior_point,
    double eps_det,
    double eps_feas) {
  if (A.size() < 3 || A.size() != b.size()) {
    return {};
  }

  std::vector<Point2> candidates;
  candidates.reserve(A.size() * 2);

  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = i + 1; j < A.size(); ++j) {
      bool ok = false;
      Point2 p = solve_line_intersection(A[i], b[i], A[j], b[j], ok, eps_det);
      if (!ok) {
        continue;
      }
      if (feasible(A, b, p, eps_feas)) {
        candidates.push_back(p);
      }
    }
  }

  if (candidates.empty()) {
    return {};
  }

  std::vector<Point2> uniq;
  uniq.reserve(candidates.size());
  for (const auto& p : candidates) {
    bool seen = false;
    for (const auto& q : uniq) {
      if (almost_same(p, q)) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      uniq.push_back(p);
    }
  }

  if (uniq.size() < 3) {
    return {};
  }

  Point2 ctr{0.0, 0.0};
  for (const auto& p : uniq) {
    ctr.x += p.x;
    ctr.y += p.y;
  }
  ctr.x /= static_cast<double>(uniq.size());
  ctr.y /= static_cast<double>(uniq.size());

  std::sort(uniq.begin(), uniq.end(), [&](const Point2& p1, const Point2& p2) {
    const double a1 = std::atan2(p1.y - ctr.y, p1.x - ctr.x);
    const double a2 = std::atan2(p2.y - ctr.y, p2.x - ctr.x);
    return a1 < a2;
  });

  if (!feasible(A, b, interior_point, eps_feas)) {
    return {};
  }

  return uniq;
}

double polygon_mask_area(const std::vector<Vec2>& A, const std::vector<double>& b, int patch_size) {
  if (A.empty() || A.size() != b.size()) {
    return 0.0;
  }

  double area = 0.0;
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x) {
      Point2 p{static_cast<double>(x), static_cast<double>(y)};
      if (feasible(A, b, p, 1e-6)) {
        area += 1.0;
      }
    }
  }
  return area;
}

double collision_mask_area(const std::vector<Vec2>& A, const std::vector<double>& b, const std::vector<uint8_t>& obs_mask, int patch_size) {
  if (A.empty() || A.size() != b.size()) {
    return 0.0;
  }

  double area = 0.0;
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x) {
      const size_t idx = static_cast<size_t>(y * patch_size + x);
      if (obs_mask[idx] == 0) {
        continue;
      }
      Point2 p{static_cast<double>(x), static_cast<double>(y)};
      if (feasible(A, b, p, 1e-6)) {
        area += 1.0;
      }
    }
  }
  return area;
}
