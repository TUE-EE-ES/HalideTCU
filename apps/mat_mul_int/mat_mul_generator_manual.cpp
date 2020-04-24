#include "Halide.h"

using namespace Halide;

namespace {

void set_alignment_and_bounds(OutputImageParam p, int size) {
  p.set_host_alignment(16).dim(0).set_bounds(0, size).dim(1).set_stride(size);
}

class MatMul : public Halide::Generator<MatMul> {
public:
  GeneratorParam<int> size{"size", 1024};

  Input<Buffer<uint8_t>> A{"A", 2};
  Input<Buffer<uint8_t>> B{"B", 2};

  Output<Buffer<int>> out{"out", 2};

  void generate() {
    Var x("x"), y("y"), p("p");

    const int warp_size = 32;
    const int vec_size = 2;
    const int x_tile = 3;
    const int y_tile = 4;
    const int y_unroll = 8;
    const int r_unroll = 1;

    Func prod("prod");
    RDom r(0, size);

    prod(x, y) += (cast<int>(A(r, y)) * cast<int>(B(x, r)));
    out(x, y) = cast<int>(prod(x, y));

    Var xi, yi, xio, xii, yii, xo, yo, x_pair, xiio, ty;
    RVar rxo, rxi;

    out.bound(x, 0, size)
        .bound(y, 0, size)
        .tile(x, y, xi, yi, x_tile * vec_size * warp_size, y_tile * y_unroll)
        .split(yi, ty, yi, y_unroll)
        .vectorize(xi, vec_size)
        .split(xi, xio, xii, warp_size)
        .reorder(xio, yi, xii, ty, x, y)
        .unroll(xio)
        .unroll(yi)
        .gpu_blocks(x, y)
        .gpu_threads(ty)
        .gpu_lanes(xii);
    prod.store_in(MemoryType::Register)
        .compute_at(out, x)
        .split(x, xo, xi, warp_size * vec_size, TailStrategy::RoundUp)
        .split(y, ty, y, y_unroll)
        .gpu_threads(ty)
        .unroll(xi, vec_size)
        .gpu_lanes(xi)
        .unroll(xo)
        .unroll(y)
        .update()
        .split(x, xo, xi, warp_size * vec_size, TailStrategy::RoundUp)
        .split(y, ty, y, y_unroll)
        .gpu_threads(ty)
        .unroll(xi, vec_size)
        .gpu_lanes(xi)
        .split(r.x, rxo, rxi, warp_size)
        .unroll(rxi, r_unroll)
        .reorder(xi, xo, y, rxi, ty, rxo)
        .unroll(xo)
        .unroll(y);

    set_alignment_and_bounds(A, size);
    set_alignment_and_bounds(B, size);
    set_alignment_and_bounds(out, size);
  }
};

} // namespace

HALIDE_REGISTER_GENERATOR(MatMul, mat_mul)
