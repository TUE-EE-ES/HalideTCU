#include <algorithm>

#include <map>

#include "Bounds.h"
#include "CSE.h"
#include "CodeGen_GPU_Dev.h"
#include "ExprUsesVar.h"
#include "IR.h"
#include "IREquality.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Simplify.h"
#include "Substitute.h"
#include <cmath>

namespace Halide {
namespace Internal {

using std::map;
using std::sort;
using std::string;
using std::vector;
using namespace std;

namespace {
string thread_names[] = {"__thread_id_x", "__thread_id_y", "__thread_id_z",
                         "__thread_id_w"};
string block_names[] = {"__block_id_x", "__block_id_y", "__block_id_z",
                        "__block_id_w"};
string shared_mem_name = "__shared";
} // namespace

class ExtractBlockSize : public IRVisitor {
  Expr block_extent[4];

  using IRVisitor::visit;

  void found_for(int dim, Expr extent) {
    internal_assert(dim >= 0 && dim < 4);
    if (!block_extent[dim].defined()) {
      block_extent[dim] = extent;
    } else {
      block_extent[dim] = simplify(Max::make(extent, block_extent[dim]));
    }
  }

  void visit(const For *op) override {
    for (int i = 0; i < 4; i++) {
      if (ends_with(op->name, thread_names[i])) {
        found_for(i, op->extent);
      }
    }

    IRVisitor::visit(op);

    Scope<Interval> scope;
    scope.push(op->name, Interval(op->min, simplify(op->min + op->extent - 1)));
    for (int i = 0; i < 4; i++) {
      if (block_extent[i].defined() &&
          expr_uses_var(block_extent[i], op->name)) {
        block_extent[i] =
            simplify(common_subexpression_elimination(block_extent[i]));
        block_extent[i] =
            simplify(bounds_of_expr_in_scope(block_extent[i], scope).max);
      }
    }
  }

  void visit(const LetStmt *op) override {
    IRVisitor::visit(op);
    for (int i = 0; i < 4; i++) {
      if (block_extent[i].defined() &&
          expr_uses_var(block_extent[i], op->name)) {
        block_extent[i] =
            simplify(Let::make(op->name, op->value, block_extent[i]));
      }
    }
  }

public:
  int dimensions() const {
    for (int i = 0; i < 4; i++) {
      if (!block_extent[i].defined()) {
        return i;
      }
    }
    return 4;
  }

  Expr extent(int d) const { return block_extent[d]; }
};

class ExtractExtents : public IRVisitor {
  Expr extents[5];
  Expr variables[5];
  std::string threads[2];
  std::string blocks[2];
  std::string kernel;
  const TensorOpDirective &tensorOp;
  using IRVisitor::visit;
  void visit(const Variable *op) override {
    Expr var = Variable::make(Int(32), op->name);
    if (ends_with(op->name, "__thread_id_x")) {
      variables[1] = var;
    } else if (ends_with(op->name, "__thread_id_y")) {

      variables[2] = var;
    } else if (ends_with(op->name, "__block_id_x")) {
      variables[3] = var;
    } else if (ends_with(op->name, "__block_id_y")) {
      variables[4] = var;
    } else if (ends_with(op->name, "$x")) {
      variables[0] = var;
    }
  }
  void visit(const For *op) override {
    if (ends_with(op->name, "__thread_id_x")) {
      threads[0] = op->name;
      extents[1] = op->extent;
    } else if (ends_with(op->name, "__thread_id_y")) {
      threads[1] = op->name;
      extents[2] = op->extent;
    } else if (ends_with(op->name, "__block_id_x")) {
      blocks[0] = op->name;
      extents[3] = op->extent;
    } else if (ends_with(op->name, "__block_id_y")) {
      blocks[1] = op->name;
      extents[4] = op->extent;
    } else {
      kernel = op->name;
      extents[0] = op->extent;
    }
    return IRVisitor::visit(op);
  }

public:
  ExtractExtents(const TensorOpDirective &tensor_op) : tensorOp(tensor_op) {}
  Expr get_arg(int arg_id) const { return extents[arg_id]; }
  Expr get_var(int arg_id) const { return variables[arg_id]; }
  std::string get_thread(int argid) const { return threads[argid]; }
  std::string get_block(int argid) const { return blocks[argid]; }
  std::string get_kernel() const { return kernel; }
};

class ExtractLoadStoreArgs : public IRVisitor {
  Expr mma_args[12];
  Type op_types[4];
  const TensorOpDirective &tensorOp;
  using IRVisitor::visit;

  void visit(const Load *op) override {
    if (op->name == tensorOp.name) {
      mma_args[4] = op->name;
      mma_args[5] = op->index;
      op_types[2] = op->type;
    }

    if (op->param.defined()) {
      if (op->param.name() == tensorOp.args[0]) {
        mma_args[0] = op->name;
        mma_args[1] = op->index;
        op_types[0] = op->type;
      } else if (op->param.name() == tensorOp.args[1]) {
        mma_args[2] = op->name;
        mma_args[3] = op->index;
        op_types[1] = op->type;
      }
    } else if (op->image.defined()) {
    }
    return IRVisitor::visit(op);
  }
  void visit(const Store *op) override {
    if (op->param.defined()) {
      mma_args[6] = op->name;
      mma_args[7] = op->index;
      op_types[3] = op->param.type();
    }

    return IRVisitor::visit(op);
  }
  void visit(const IfThenElse *op) override {
    if (op->condition.defined())
      mma_args[8] = simplify(op->condition);

    return IRVisitor::visit(op);
  }

public:
  ExtractLoadStoreArgs(const TensorOpDirective &tensor_op)
      : tensorOp(tensor_op) {}
  Expr get_arg(int arg_id) const { return mma_args[arg_id]; }
  Type get_type(int arg_id) const { return op_types[arg_id]; }
};

class FixLoadsStores : public IRMutator {
  const TensorOpDirective &tensorOp;
  const ExtractExtents &extents;
  using IRMutator::visit;
  const bool fix_stores;
  const bool is_loadC;
  const bool is_mma;
  Expr blockx = extents.get_var(3);
  Expr threadx = extents.get_var(1);
  Expr kernel = extents.get_var(0);
  Expr wsize = make_const(Int(32), 32);
  Expr ldc = make_const(Int(32), 2048);
  Expr visit(const Load *op) override {
    if (op->param.defined() && op->param.name() == tensorOp.args[0]) {
      Expr new_index = op->index;
      return Load::make(op->type, op->name, new_index, Buffer<>(), op->param,
                        op->predicate, op->alignment);
    } else if (op->param.defined() && op->param.name() == tensorOp.args[1]) {
      Expr new_index = op->index;
      return Load::make(op->type, op->name, new_index, Buffer<>(), op->param,
                        op->predicate, op->alignment);
    } else if ((op->name == tensorOp.name) && (is_loadC)) {
      Expr new_index = op->index;
      return Load::make(op->type, op->name, new_index, Buffer<>(), op->param,
                        op->predicate, op->alignment);
    } else {
      return make_zero(op->type);
    }
    return IRMutator::visit(op);
  }
  Stmt visit(const Store *op) override {
    Expr new_value = op->value;
    std::string new_name;
    if (is_loadC)
      new_name = "loadC";
    else if (is_mma)
      new_name = "mma";
    new_value = mutate(op->value);
    if (op->param.name() == tensorOp.name) {
      Expr new_index = op->index;
      Stmt new_body;
      if (fix_stores)
        new_body = Store::make(op->name, new_value, new_index, op->param,
                               op->predicate, op->alignment);
      else
        new_body = Store::make(op->name, new_value, new_index, op->param,
                               op->predicate, op->alignment);
      return IfThenElse::make(op->predicate, new_body, Stmt());
    } else
      return IRMutator::visit(op);
  }

public:
  FixLoadsStores(const TensorOpDirective &tensor_op, const ExtractExtents &exs,
                 const bool &fxs, const bool &is_load, const bool &ismac)
      : tensorOp(tensor_op), extents(exs), fix_stores(fxs), is_loadC(is_load),
        is_mma(ismac) {}
};

class InjectTensorBlocks : public IRMutator {
  using IRMutator::visit;
  const ExtractBlockSize &block_size;
  const TensorOpDirective &tensorOp;
  const ExtractExtents &extents;

  std::vector<Stmt> prev_block;
  Stmt inner_block;

  Stmt visit(const For *op) override {

    if (op->for_type == ForType::Serial) {
      Stmt body = op->body;
      FixLoadsStores fx(tensorOp, extents, true, true, true);
      body = fx.mutate(body);
      ExtractLoadStoreArgs exLSA(tensorOp);

      body.accept(&exLSA);

      std::vector<Expr> arguments;
      Expr rvar = Variable::make(Int(32), op->name);

      arguments.push_back(rvar);
      arguments.push_back(op->min);
      arguments.push_back(op->extent);
      Expr blockx = extents.get_var(3);
      Expr threadx = extents.get_var(1);
      Expr kernel = extents.get_var(0);
      arguments.push_back(threadx);
      arguments.push_back(blockx);
      arguments.push_back(extents.get_arg(1));
      arguments.push_back(exLSA.get_arg(4));
      arguments.push_back(exLSA.get_arg(0));
      arguments.push_back(exLSA.get_arg(2));
      std::vector<Type> typez;
      for (int i = 0; i < 4; i++) {
        typez.push_back(exLSA.get_type(i));
      }
      Stmt for_body = For::make(op->name, op->min, op->extent, op->for_type,
                                op->device_api, body);
      Stmt tensor_operation_stmt =
          TensorOp::make("TensorOperation", typez, arguments, for_body);
      body = tensor_operation_stmt;
      return body;
    }

    else
      return IRMutator::visit(op);
  }

public:
  InjectTensorBlocks(const ExtractBlockSize &bs,
                     const TensorOpDirective &tensor_op,
                     const ExtractExtents &exs)
      : block_size(bs), tensorOp(tensor_op), extents(exs) {}
};

class RemoveDim : public IRMutator {

  using IRMutator::visit;
  TensorOpDirective &tensor_op;
  Stmt visit(const For *op) override {
    if (ends_with(op->name, "s1.y")) {
      Stmt body = op->body;
      Expr new_var = make_one(Int(32));
      body = substitute(op->name, new_var, body);
      return body;
    } else
      return IRMutator::visit(op);
  }

public:
  RemoveDim(TensorOpDirective tensorOp) : tensor_op(tensorOp) {}
};
class FixExtent : public IRMutator {

  using IRMutator::visit;
  TensorOpDirective &tensor_op;
  Stmt visit(const For *op) override {
    if (ends_with(op->name, "__block_id_x")) {
      RemoveDim rf(tensor_op);
      Stmt body = rf.mutate(op->body);
      return For::make(op->name, op->min, make_const(Int(32), 68), op->for_type,
                       op->device_api, body);
    } else
      return IRMutator::visit(op);
  }

public:
  FixExtent(TensorOpDirective tensorOp) : tensor_op(tensorOp) {}
};
class InjectTensorOps : public IRMutator {
public:
  InjectTensorOps(TensorOpDirective tensorOp) : tensor_op(tensorOp) {}
  using IRMutator::visit;
  string prefix;
  TensorOpDirective &tensor_op;

  Stmt visit(const For *op) override {

    prefix = tensor_op.name;

    if (CodeGen_GPU_Dev::is_gpu_block_var(op->name) &&
        starts_with(op->name, prefix) &&
        !starts_with(op->name, prefix + ".s0")) {
      FixExtent fx(tensor_op);

      ExtractBlockSize block_size;

      Stmt body = fx.mutate(op);

      Stmt loop = body;
      loop.accept(&block_size);
      ExtractExtents exs(tensor_op);
      loop.accept(&exs);
      return InjectTensorBlocks(block_size, tensor_op, exs).mutate(loop);
    } else {
      return IRMutator::visit(op);
    }
  }
};

Stmt inject_tensor_ops(Stmt s, map<string, Function> env) {
  // find all tensor ops!
  for (const auto &st : env) {
    const Function &f = st.second;
    for (int i = 0; i < (int)(f.updates().size()); i++) {
      Definition def = f.update(i);
      if (def.schedule().tensor_core()) {
        TensorOpDirective tensor_op = def.schedule().tensorOp();
        s = InjectTensorOps(tensor_op).mutate(s);
      }
    }
  }
  return s;
}

} // namespace Internal
} // namespace Halide
