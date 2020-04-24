#ifndef INJECT_TENSOR_OPS_H
#define INJECT_TENSOR_OPS_H

/** \file
 * Wraps Funcs that need to be mapped on a TCU around a TensorOp node.
 * Also extracts a bunch of extents/param names to be used in the TensorOp for
 * NVVM codegen later. Fixes loop bounds for block dim (CTA outer dim)
 */

#include "IR.h"

namespace Halide {
namespace Internal {

Stmt inject_tensor_ops(Stmt s, std::map<std::string, Function> env);

} // namespace Internal
} // namespace Halide

#endif
