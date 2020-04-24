#ifndef LOWER_TOPS_H
#define LOWER_TOPS_H

/** \file
 * Defines the lowering pass that fuses and normalizes loops over gpu
 * threads to target CUDA, OpenCL, and Metal.
 */

#include "IR.h"

namespace Halide {
namespace Internal {

/** Rewrite all GPU loops to have a min of zero. */
// Stmt zero_gpu_loop_mins(Stmt s);

/** Converts Halide's GPGPU IR to the OpenCL/CUDA/Metal model. Within every
 * loop over gpu block indices, fuse the inner loops over thread
 * indices into a single loop (with predication to turn off
 * threads). Also injects synchronization points as needed, and hoists
 * allocations at the block level out into a single shared memory
 * array. */
Stmt lower_tensor_body(const TensorOp *op);

} // namespace Internal
} // namespace Halide

#endif
