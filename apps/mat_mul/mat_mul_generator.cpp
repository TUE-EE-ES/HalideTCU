#include "Halide.h"
using namespace Halide;
namespace {

class MatMul : public Halide::Generator<MatMul> {
public:
  GeneratorParam<int> matrix_size{"matrix_size", 1024};

  Input<Buffer<float16_t>> input_a{"input_a", 2};
  Input<Buffer<float16_t>> input_b{"input_b", 2};
  Output<Buffer<float>> matrix_mul{"output", 2};
  GeneratorParam<bool> use_tensor{"use_tensor", false};
  void generate() {
    Var x("x"), y("y");
    // Algorithm
    RDom k(0, matrix_size);

    matrix_mul(x, y) = undef<float>();
    matrix_mul(x, y) +=
        cast<float>(cast<float>(input_a(k, y)) * cast<float>(input_b(x, k)));

    // Schedule
    if (!auto_schedule) {
      if (get_target().has_gpu_feature()) {
        if (use_tensor) {
          Var xi("xi"), yi("yi"), xii("xii"), yii("yii"), xt("xt"), yt("yt");
          RVar ki("ki");
          matrix_mul.compute_root().gpu_tile(x, y, xi, yi, 16, 16);
          matrix_mul.update().tensor_core(input_a, input_b);
          matrix_mul.bound(x, 0, matrix_size).bound(y, 0, matrix_size);

        } else {
          Var xi("xi"), yi("yi"), xii("xii"), yii("yii"), xt("xt"), yt("yt");
          RVar ki("ki");
          matrix_mul.compute_root().gpu_tile(x, y, xi, yi, 32, 16);
          matrix_mul.update().gpu_tile(x, y, xi, yi, 32, 16);
          matrix_mul.bound(x, 0, matrix_size).bound(y, 0, matrix_size);
        }

      } else {

        Var xi("xi"), yi("yi"), yii("yii"), xii("xii"), xy("xy");
        matrix_mul.tile(x, y, xi, yi, 24, 32)
            .fuse(x, y, xy)
            .parallel(xy)
            .split(yi, yi, yii, 4)
            .vectorize(xi, 8)
            .unroll(xi)
            .unroll(yii);
      }
    }

    // Always specify bounds for outputs, whether autoscheduled or not
    // Estimates
    {
      input_a.dim(0)
          .set_estimate(0, matrix_size)
          .dim(1)
          .set_estimate(0, matrix_size);
      input_b.dim(0)
          .set_estimate(0, matrix_size)
          .dim(1)
          .set_estimate(0, matrix_size);
    }
  }
};

} // namespace

HALIDE_REGISTER_GENERATOR(MatMul, mat_mul)
