#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "geometry2d.hpp"

namespace {

bool load_mask_txt(const std::string& path, int patch_size, std::vector<uint8_t>& mask) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    return false;
  }

  mask.assign(static_cast<size_t>(patch_size * patch_size), 0);
  std::string line;
  for (int y = 0; y < patch_size; ++y) {
    if (!std::getline(fin, line)) {
      return false;
    }

    int xw = 0;
    for (char ch : line) {
      if (ch != '0' && ch != '1') {
        continue;
      }
      if (xw >= patch_size) {
        break;
      }
      mask[static_cast<size_t>(y * patch_size + xw)] = (ch == '1') ? 1 : 0;
      ++xw;
    }
    if (xw != patch_size) {
      return false;
    }
  }

  return true;
}

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

std::string get_arg(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      return argv[i + 1];
    }
  }
  return fallback;
}

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string& s) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  return conv.from_bytes(s);
}
#endif

double get_arg_double(int argc, char** argv, const std::string& key, double fallback) {
  std::string v = get_arg(argc, argv, key, "");
  if (v.empty()) {
    return fallback;
  }
  return std::stod(v);
}

int get_arg_int(int argc, char** argv, const std::string& key, int fallback) {
  std::string v = get_arg(argc, argv, key, "");
  if (v.empty()) {
    return fallback;
  }
  return std::stoi(v);
}

}  // namespace

int main(int argc, char** argv) {
  try {
  const std::string onnx_path = get_arg(argc, argv, "--onnx", "models/neural_iris_net.onnx");
  const std::string mask_path = get_arg(argc, argv, "--mask", "");
  const int patch_size = get_arg_int(argc, argv, "--patch", 128);
  const double safety_margin = get_arg_double(argc, argv, "--safety", 0.5);

  if (mask_path.empty()) {
    std::cerr << "Usage: neural_iris_cpp_infer --onnx <model.onnx> --mask <mask.txt> [--patch 128] [--safety 0.5]" << std::endl;
    return 1;
  }

  std::cout << "onnx: " << onnx_path << "\n";
  std::cout << "mask: " << mask_path << "\n";

  std::vector<uint8_t> obs_mask;
  if (!load_mask_txt(mask_path, patch_size, obs_mask)) {
    std::cerr << "Failed to load mask: " << mask_path << std::endl;
    return 1;
  }

  std::vector<float> input(static_cast<size_t>(patch_size * patch_size), 0.0f);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(obs_mask[i]);
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "neural_iris_cpp");
  Ort::SessionOptions session_opts;
  session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA_PROVIDER
  OrtCUDAProviderOptions cuda_options{};
  session_opts.AppendExecutionProvider_CUDA(cuda_options);
#endif

  #ifdef _WIN32
  std::wstring onnx_path_w = utf8_to_wstring(onnx_path);
  Ort::Session session(env, onnx_path_w.c_str(), session_opts);
  #else
  Ort::Session session(env, onnx_path.c_str(), session_opts);
  #endif

  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
  auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
  const char* input_name = input_name_alloc.get();
  const char* output_name = output_name_alloc.get();

  std::array<int64_t, 4> input_shape = {1, 1, patch_size, patch_size};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info,
      input.data(),
      input.size(),
      input_shape.data(),
      input_shape.size());

  auto t1 = std::chrono::high_resolution_clock::now();
  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr},
      &input_name,
      &input_tensor,
      1,
      &output_name,
      1);
  auto t2 = std::chrono::high_resolution_clock::now();

  float* pred = output_tensors[0].GetTensorMutableData<float>();

  Mat2 P{};
  Point2 c{};
  parse_pred_to_geometry(pred, patch_size, P, c);

  auto t3 = std::chrono::high_resolution_clock::now();
  std::vector<Point2> obs_points = extract_obstacle_boundary(obs_mask, patch_size, patch_size);

  std::vector<Vec2> A;
  std::vector<double> b;
  build_safe_halfspaces(P, c, obs_points, patch_size, safety_margin, A, b);

  Point2 center{patch_size / 2.0, patch_size / 2.0};
  std::vector<Point2> poly = halfspace_intersection_2d(A, b, center);
  auto t4 = std::chrono::high_resolution_clock::now();

  const double poly_area = polygon_mask_area(A, b, patch_size);
  const double coll_area = collision_mask_area(A, b, obs_mask, patch_size);
  const double coll_ratio = (poly_area > 1e-9) ? (coll_area / poly_area) : 0.0;

  auto dt_setup = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto dt_forward = std::chrono::duration<double, std::milli>(t2 - t1).count();
  auto dt_geom = std::chrono::duration<double, std::milli>(t4 - t3).count();
  auto dt_total = std::chrono::duration<double, std::milli>(t4 - t0).count();

  std::cout << "pred: ["
            << pred[0] << ", " << pred[1] << ", " << pred[2] << ", "
            << pred[3] << ", " << pred[4] << ", " << pred[5] << "]\n";
  std::cout << "obs_boundary_points: " << obs_points.size() << "\n";
  std::cout << "halfspaces: " << A.size() << "\n";
  std::cout << "polygon_vertices: " << poly.size() << "\n";
  std::cout << "collision_ratio: " << (100.0 * coll_ratio) << "%\n";

  std::cout << "timing_ms: setup=" << dt_setup
            << " forward=" << dt_forward
            << " geometry=" << dt_geom
            << " total=" << dt_total << "\n";

  return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "[ORT ERROR] " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "[STD ERROR] " << e.what() << "\n";
    return 3;
  } catch (...) {
    std::cerr << "[UNKNOWN ERROR]" << "\n";
    return 4;
  }
}


