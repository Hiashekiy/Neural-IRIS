#include <array>
#include <cstdint>
#include <codecvt>
#include <locale>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "geometry2d.hpp"

namespace {

std::mutex g_mu;
std::unique_ptr<Ort::Env> g_env;
std::unique_ptr<Ort::Session> g_sess;
std::string g_input_name;
std::string g_output_name;

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string& s) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  return conv.from_bytes(s);
}
#endif

void parse_pred_to_geometry(const float* pred, int patch_size, Mat2& P, Point2& c) {
  const double dx = static_cast<double>(pred[0]);
  const double dy = static_cast<double>(pred[1]);
  const double a = static_cast<double>(pred[2]);
  const double b = static_cast<double>(pred[3]);
  const double sin_t = static_cast<double>(pred[4]);
  const double cos_t = static_cast<double>(pred[5]);

  c.x = patch_size / 2.0 + dx;
  c.y = patch_size / 2.0 + dy;

  const double inv_a = 1.0 / (a + 1e-4);
  const double inv_b = 1.0 / (b + 1e-4);

  const Mat2 R = {{{cos_t, -sin_t}, {sin_t, cos_t}}};
  const Mat2 L = {{{inv_a, 0.0}, {0.0, inv_b}}};

  Mat2 RL{};
  RL[0][0] = R[0][0] * L[0][0] + R[0][1] * L[1][0];
  RL[0][1] = R[0][0] * L[0][1] + R[0][1] * L[1][1];
  RL[1][0] = R[1][0] * L[0][0] + R[1][1] * L[1][0];
  RL[1][1] = R[1][0] * L[0][1] + R[1][1] * L[1][1];

  P[0][0] = RL[0][0] * R[0][0] + RL[0][1] * R[0][1];
  P[0][1] = RL[0][0] * R[1][0] + RL[0][1] * R[1][1];
  P[1][0] = RL[1][0] * R[0][0] + RL[1][1] * R[0][1];
  P[1][1] = RL[1][0] * R[1][0] + RL[1][1] * R[1][1];
}

}  // namespace

extern "C" {

__declspec(dllexport) int corridor_init(const char* model_path_utf8) {
  try {
    std::lock_guard<std::mutex> lk(g_mu);

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  #ifdef USE_CUDA_PROVIDER
    OrtCUDAProviderOptions cuda_options{};
    opts.AppendExecutionProvider_CUDA(cuda_options);
  #endif

    g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "corridor_cpp_bridge");

#ifdef _WIN32
    std::wstring model_path_w = utf8_to_wstring(std::string(model_path_utf8));
    g_sess = std::make_unique<Ort::Session>(*g_env, model_path_w.c_str(), opts);
#else
    g_sess = std::make_unique<Ort::Session>(*g_env, model_path_utf8, opts);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = g_sess->GetInputNameAllocated(0, allocator);
    auto output_name_alloc = g_sess->GetOutputNameAllocated(0, allocator);
    g_input_name = input_name_alloc.get();
    g_output_name = output_name_alloc.get();

    return 0;
  } catch (...) {
    g_sess.reset();
    g_env.reset();
    return 1;
  }
}

__declspec(dllexport) int corridor_infer(
    const uint8_t* mask_ptr,
    int patch_size,
    double safety_margin,
    double* out_vertices_xy,
    int max_vertices,
    int* out_vertex_count) {
  try {
    if (!g_sess || !mask_ptr || !out_vertices_xy || !out_vertex_count || patch_size <= 0 || max_vertices <= 0) {
      return 2;
    }

    std::lock_guard<std::mutex> lk(g_mu);

    const size_t n = static_cast<size_t>(patch_size * patch_size);
    std::vector<float> input(n, 0.0f);
    std::vector<uint8_t> obs_mask(n, 0);
    for (size_t i = 0; i < n; ++i) {
      obs_mask[i] = mask_ptr[i] ? 1 : 0;
      input[i] = static_cast<float>(obs_mask[i]);
    }

    std::array<int64_t, 4> input_shape = {1, 1, patch_size, patch_size};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size());

    const char* in_name = g_input_name.c_str();
    const char* out_name = g_output_name.c_str();
    auto output_tensors = g_sess->Run(Ort::RunOptions{nullptr}, &in_name, &input_tensor, 1, &out_name, 1);

    float* pred = output_tensors[0].GetTensorMutableData<float>();

    Mat2 P{};
    Point2 c{};
    parse_pred_to_geometry(pred, patch_size, P, c);

    std::vector<Point2> obs_points = extract_obstacle_boundary(obs_mask, patch_size, patch_size);
    std::vector<Vec2> A;
    std::vector<double> b;
    build_safe_halfspaces(P, c, obs_points, patch_size, safety_margin, A, b);

    Point2 center{patch_size / 2.0, patch_size / 2.0};
    std::vector<Point2> poly = halfspace_intersection_2d(A, b, center);
    if (poly.size() < 3) {
      poly = halfspace_intersection_2d(A, b, c);
    }

    const int write_n = static_cast<int>(std::min<size_t>(poly.size(), static_cast<size_t>(max_vertices)));
    for (int i = 0; i < write_n; ++i) {
      out_vertices_xy[2 * i + 0] = poly[static_cast<size_t>(i)].x;
      out_vertices_xy[2 * i + 1] = poly[static_cast<size_t>(i)].y;
    }
    *out_vertex_count = write_n;
    return 0;
  } catch (...) {
    *out_vertex_count = 0;
    return 3;
  }
}

__declspec(dllexport) int corridor_infer_batch(
    const uint8_t* masks_ptr,
    int batch_size,
    int patch_size,
    double safety_margin,
    double* out_vertices_xy,
    int max_vertices,
    int* out_vertex_counts) {
  try {
    if (!g_sess || !masks_ptr || !out_vertices_xy || !out_vertex_counts ||
        batch_size <= 0 || patch_size <= 0 || max_vertices <= 0) {
      return 2;
    }

    std::lock_guard<std::mutex> lk(g_mu);

    const size_t one_n = static_cast<size_t>(patch_size * patch_size);
    const size_t all_n = static_cast<size_t>(batch_size) * one_n;

    std::vector<float> input(all_n, 0.0f);
    for (size_t i = 0; i < all_n; ++i) {
      input[i] = masks_ptr[i] ? 1.0f : 0.0f;
    }

    std::array<int64_t, 4> input_shape = {batch_size, 1, patch_size, patch_size};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size());

    const char* in_name = g_input_name.c_str();
    const char* out_name = g_output_name.c_str();
    auto output_tensors = g_sess->Run(Ort::RunOptions{nullptr}, &in_name, &input_tensor, 1, &out_name, 1);

    float* pred = output_tensors[0].GetTensorMutableData<float>();

    for (int bi = 0; bi < batch_size; ++bi) {
      out_vertex_counts[bi] = 0;
      const float* pred_i = pred + static_cast<size_t>(bi) * 6;

      Mat2 P{};
      Point2 c{};
      parse_pred_to_geometry(pred_i, patch_size, P, c);

      const uint8_t* mask_i_ptr = masks_ptr + static_cast<size_t>(bi) * one_n;
      std::vector<uint8_t> obs_mask(one_n, 0);
      for (size_t k = 0; k < one_n; ++k) {
        obs_mask[k] = mask_i_ptr[k] ? 1 : 0;
      }

      std::vector<Point2> obs_points = extract_obstacle_boundary(obs_mask, patch_size, patch_size);
      std::vector<Vec2> A;
      std::vector<double> b;
      build_safe_halfspaces(P, c, obs_points, patch_size, safety_margin, A, b);

      Point2 center{patch_size / 2.0, patch_size / 2.0};
      std::vector<Point2> poly = halfspace_intersection_2d(A, b, center);
      if (poly.size() < 3) {
        poly = halfspace_intersection_2d(A, b, c);
      }

      const int write_n = static_cast<int>(std::min<size_t>(poly.size(), static_cast<size_t>(max_vertices)));
      out_vertex_counts[bi] = write_n;

      const size_t out_off = static_cast<size_t>(bi) * static_cast<size_t>(max_vertices) * 2;
      for (int vi = 0; vi < write_n; ++vi) {
        out_vertices_xy[out_off + static_cast<size_t>(2 * vi + 0)] = poly[static_cast<size_t>(vi)].x;
        out_vertices_xy[out_off + static_cast<size_t>(2 * vi + 1)] = poly[static_cast<size_t>(vi)].y;
      }
    }

    return 0;
  } catch (...) {
    for (int bi = 0; bi < batch_size; ++bi) {
      out_vertex_counts[bi] = 0;
    }
    return 3;
  }
}

__declspec(dllexport) void corridor_shutdown() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_sess.reset();
  g_env.reset();
  g_input_name.clear();
  g_output_name.clear();
}

}
