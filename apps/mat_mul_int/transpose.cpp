#include "Halide.h"
// a#include "../autoscheduler/SimpleAutoSchedule.h"
using namespace Halide;
namespace {

class Transpose : public Halide::Generator<Transpose> {
public:
  GeneratorParam<int> matrix_size{"matrix_size", 1024};

  Input<Buffer<uint8_t>> input{"input_a", 2};
  Output<Buffer<uint8_t>> matrix_mul{"output", 2};
  void generate() {
    Var x("x"), y("y");

    // Algorithm
    RDom k(0, matrix_size);

    Func block("block");
    block(x, y) = input(x, y);
    matrix_mul(x, y) = block(y, x);

    if (get_target().has_gpu_feature()) {
      Var xi, yi, xo, yo, xii, xio, yii, yio, tile_idx, subtile_idx;
      matrix_mul.tile(x, y, xo, yo, xi, yi, 64, 64)
          .fuse(xo, yo, tile_idx)
          .tile(xi, yi, xio, yio, xii, yii, 16, 16)
          .fuse(xio, yio, subtile_idx)
          .gpu_blocks(subtile_idx, tile_idx)
          .gpu_threads(xii, yii);

      // Load a tile on input and store it into shared.
      block.compute_at(matrix_mul, subtile_idx).gpu_threads(x, y);
      matrix_mul.bound(x, 0, matrix_size).bound(y, 0, matrix_size);
      input.set_host_alignment(16)
          .dim(0)
          .set_bounds(0, matrix_size)
          .dim(1)
          .set_stride(matrix_size);
    }
  }
};

} // namespace

HALIDE_REGISTER_GENERATOR(Transpose, transpose)
