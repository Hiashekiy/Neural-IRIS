#include <decomp_geometry/geometric_utils.h>
#include <decomp_util/seed_decomp.h>

#include <cstdlib>
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

  vec_Vec2f obs;
  obs.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    double x = 0.0;
    double y = 0.0;
    if (!(std::cin >> x >> y)) {
      std::cout << "ERR invalid_obs_point" << std::endl;
      return 1;
    }
    obs.push_back(Vec2f(x, y));
  }

  const Vec2f center(cx, cy);

  SeedDecomp2D decomp(center);
  decomp.set_local_bbox(Vec2f(patch_size, patch_size));
  decomp.set_obs(obs);

  // Use a small inflation radius in pixel space; walls and obstacles define the final shape.
  decomp.dilate(0.5);

  const auto poly = decomp.get_polyhedron();
  auto vertices = cal_vertices(poly);

  if (vertices.size() < 3) {
    std::cout << "ERR no_polygon" << std::endl;
    return 2;
  }

  std::cout << "OK " << vertices.size() << std::endl;
  for (const auto& v : vertices) {
    std::cout << v(0) << " " << v(1) << std::endl;
  }

  return 0;
}
