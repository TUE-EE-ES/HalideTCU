#include "CodeGen_PTX_Dev.h"
#include "CSE.h"
#include "CodeGen_Internal.h"
#include "Debug.h"
#include "ExprUsesVar.h"
#include "IREquality.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "LLVM_Headers.h"
#include "LLVM_Runtime_Linker.h"
#include "Simplify.h"
#include "Solve.h"
#include "Target.h"
#include "UnrollLoops.h"
#include "top_equiv.h"
#include <fstream>
#include <string>
// This is declared in NVPTX.h, which is not exported. Ugly, but seems better
// than hardcoding a path to the .h file.
#ifdef WITH_PTX
namespace llvm {
FunctionPass *createNVVMReflectPass(const StringMap<int> &Mapping);
}
#endif

namespace Halide {
namespace Internal {

using std::string;
using std::vector;

using namespace llvm;

string get_expr_str(Expr expr) {
  std::ostringstream ostr;
  ostr << expr;
  string nst;
  nst = ostr.str();
  nst.erase(std::remove(nst.begin(), nst.end(), '"'), nst.end());
  return nst;
}

CodeGen_PTX_Dev::CodeGen_PTX_Dev(Target host) : CodeGen_LLVM(host) {
#if !defined(WITH_PTX)
  user_error << "ptx not enabled for this build of Halide.\n";
#endif
  user_assert(llvm_NVPTX_enabled)
      << "llvm build not configured with nvptx target enabled\n.";

  context = new llvm::LLVMContext();
}

CodeGen_PTX_Dev::~CodeGen_PTX_Dev() {
  // This is required as destroying the context before the module
  // results in a crash. Really, responsibility for destruction
  // should be entirely in the parent class.
  // TODO: Figure out how to better manage the context -- e.g. allow using
  // same one as the host.
  module.reset();
  delete context;
}

Type CodeGen_PTX_Dev::upgrade_type_for_storage(const Type &t) const {
  if (t.element_of() == Float(16))
    return t;
  return CodeGen_LLVM::upgrade_type_for_storage(t);
}

void CodeGen_PTX_Dev::add_kernel(Stmt stmt, const std::string &name,
                                 const std::vector<DeviceArgument> &args) {
  internal_assert(module != nullptr);

  debug(1) << "In CodeGen_PTX_Dev::add_kernel\n";

  // Now deduce the types of the arguments to our function
  vector<llvm::Type *> arg_types(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].is_buffer) {
      arg_types[i] = llvm_type_of(UInt(8))->getPointerTo();
    } else {
      arg_types[i] = llvm_type_of(args[i].type);
    }
  }

  // Make our function
  FunctionType *func_t = FunctionType::get(void_t, arg_types, false);
  function = llvm::Function::Create(func_t, llvm::Function::ExternalLinkage,
                                    name, module.get());
  set_function_attributes_for_target(function, target);

  // Mark the buffer args as no alias
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].is_buffer) {
      function->addParamAttr(i, Attribute::NoAlias);
    }
  }

  // Make the initial basic block
  entry_block = BasicBlock::Create(*context, "entry", function);
  builder->SetInsertPoint(entry_block);
  // Put the arguments in the symbol table
  vector<string> arg_sym_names;
  {
    size_t i = 0;
    for (auto &fn_arg : function->args()) {

      string arg_sym_name = args[i].name;
      sym_push(arg_sym_name, &fn_arg);
      fn_arg.setName(arg_sym_name);
      arg_sym_names.push_back(arg_sym_name);

      i++;
    }
  }

  // We won't end the entry block yet, because we'll want to add
  // some allocas to it later if there are local allocations. Start
  // a new block to put all the code.
  BasicBlock *body_block = BasicBlock::Create(*context, "body", function);
  builder->SetInsertPoint(body_block);

  debug(1) << "Generating llvm bitcode for kernel...\n";
  // Ok, we have a module, function, context, and a builder
  // pointing at a brand new basic block. We're good to go.
  stmt.accept(this);

  // Now we need to end the function
  builder->CreateRetVoid();

  // Make the entry block point to the body block
  builder->SetInsertPoint(entry_block);
  builder->CreateBr(body_block);

  // Add the nvvm annotation that it is a kernel function.
  llvm::Metadata *md_args[] = {
      llvm::ValueAsMetadata::get(function), MDString::get(*context, "kernel"),
      llvm::ValueAsMetadata::get(ConstantInt::get(i32_t, 1))};

  MDNode *md_node = MDNode::get(*context, md_args);

  module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);

  // Now verify the function is ok
  verifyFunction(*function);

  // Finally, verify the module is ok
  verifyModule(*module);

  debug(2) << "Done generating llvm bitcode for PTX\n";

  // Clear the symbol table
  for (size_t i = 0; i < arg_sym_names.size(); i++) {
    sym_pop(arg_sym_names[i]);
  }
}

void CodeGen_PTX_Dev::init_module() {
  init_context();

#ifdef WITH_PTX
  module = get_initial_module_for_ptx_device(target, context);
#endif
}

void CodeGen_PTX_Dev::visit(const Call *op) {

  if (op->is_intrinsic(Call::mma_memcpy_C_to_shared)) {

    Expr Load_index = op->args[0];
    Expr Store_index = op->args[1];
    Expr laneID = op->args[2];
    Value *idx = codegen(laneID);
    Value *from = ConstantInt::get(i32_t, 0);
    Value *sh_base = codegen_buffer_pointer("__shared", op->type, Store_index);
    Value *to = upgrade_for_memcpy(sh_base, i4_t, idx);

    CallInst *mc1 = builder->CreateMemSet(to, from, 16, 16, false);

    add_tbaa_metadata(mc1, "fragment_c", 16);
    ;

  } else if (op->is_intrinsic(Call::mma_memcpy_C_to_global)) {
    Expr Load_index = op->args[0];
    Value *sh_base = codegen_buffer_pointer("__shared", op->type, Load_index);
    Expr laneID = op->args[2];
    Value *idx = codegen(laneID);
    Value *from = upgrade_for_memcpy(sh_base, i4_t, idx);
    Expr Store_index = op->args[1];
    Value *c_base = codegen_buffer_pointer(get_expr_str(op->args[3]), op->type,
                                           Store_index);
    Value *to = upgrade_for_memcpy(c_base, i4_t, idx);

    CallInst *mc2 = builder->CreateMemCpy(to, 16, from, 16, 16, false);
    add_tbaa_metadata(mc2, "fragment_c", 16);
    ;
  } else if (op->is_intrinsic(Call::mma_memcpy_AB)) {
    Expr warpID = op->args[0];
    Expr input_a = op->args[1];
    Expr input_b = op->args[2];
    Expr offset_e = op->args[3];
    Expr cond = warpID < 4;
    Expr indxA = op->args[4];
    Expr indxB = op->args[5];
    Expr memcpy_offset = op->args[6];
    Expr Store_index_l = op->args[7];
    Value *cond_wptr = codegen(cond);
    Value *Bwarp_ptr =
        codegen_buffer_pointer(get_expr_str(input_b), op->type, indxB);
    Value *Awarp_ptr =
        codegen_buffer_pointer(get_expr_str(input_a), op->type, indxA);
    ;
    Value *idx_l = codegen(memcpy_offset);
    Value *warp_ptr = builder->CreateSelect(cond_wptr, Awarp_ptr, Bwarp_ptr);
    Value *offset_v = codegen(offset_e);
    warp_ptr = builder->CreateInBoundsGEP(warp_ptr, offset_v);
    Value *lane_ptr = upgrade_for_memcpy(warp_ptr, i4_t, idx_l);
    Value *idx_s = codegen(memcpy_offset);
    Value *sh_base_s =
        codegen_buffer_pointer("__shared", op->type, Store_index_l);
    sh_base_s = upgrade_for_memcpy(sh_base_s, i4_t, idx_s);

    CallInst *mc3 =
        builder->CreateMemCpy(sh_base_s, 16, lane_ptr, 16, 16, false);
    add_tbaa_metadata(mc3, "fragment_c", 16);

  }

  else if (op->is_intrinsic(Call::mma_load)) {
    std::string loadintr;
    std::string fname;
    if (can_prove(op->args[0] == 0)) {
      // std::cout<<"load A "<<std::endl;
      fname = "fragment_a" + get_expr_str(op->args[4]);
      if (get_expr_str(op->args[6]) == "row")
        loadintr = mma_load_intrinsic(op->type, 16, 16, 16, a, row);
      else if (get_expr_str(op->args[6]) == "col")
        loadintr = mma_load_intrinsic(op->type, 16, 16, 16, a, col);

    } else if (can_prove(op->args[0] == 1)) {
      fname = "fragment_b" + get_expr_str(op->args[5]);
      if (get_expr_str(op->args[6]) == "row")
        loadintr = mma_load_intrinsic(op->type, 16, 16, 16, b, row);
      else if (get_expr_str(op->args[6]) == "col")
        loadintr = mma_load_intrinsic(op->type, 16, 16, 16, b, col);
    } else if (can_prove(op->args[0] == 2)) {
      fname = "fragment_c";
      loadintr = mma_load_intrinsic(op->type, 16, 16, 16, c, row);
    }
    // codegen the index
    llvm::Function *fnL = module->getFunction(loadintr);
    Value *idx = codegen(op->args[2]);
    Value *ptr = nullptr;
    if (can_prove(op->args[0] == 2))
      ptr = codegen_fragment_pointer(get_expr_str(op->args[1]), op->type,
                                     op->type, idx, 8);
    else
      ptr = codegen_fragment_pointer(get_expr_str(op->args[1]), Int(32),
                                     op->type, idx, 8);
    Value *stride = codegen(op->args[3]);
    Value *fragment = builder->CreateCall(fnL, {ptr, stride});

    // if it's load c we need to store the fragment
    if (can_prove(op->args[0] == 2)) {
      Value *fragment_c = codegen_buffer_pointer(
          "fragment_c" + get_expr_str(op->args[4]) + get_expr_str(op->args[5]),
          op->type, make_zero(Int(32)));
      for (size_t i = 0; i < 8; i++) {
        Value *p = builder->CreateExtractValue(fragment, i);
        Value *idx = ConstantInt::get(i64_t, i);
        Value *elem_ptr = builder->CreateInBoundsGEP(fragment_c, idx);
        StoreInst *val = builder->CreateStore(p, elem_ptr);
        Expr ind = make_const(Int(32), i);
        add_tbaa_metadata(val, "fragment_c", ind);
        ;
      }
    } else
      sym_push(fname, fragment);

    value = ConstantInt::get(i32_t, 0);
    // call
  } else if (op->is_intrinsic(Call::mma_operation)) {
    Value *fragment_a = sym_get("fragment_a" + get_expr_str(op->args[0]));
    Value *fragment_b = sym_get("fragment_b" + get_expr_str(op->args[1]));
    int fragA_size = (op->type == Float(32)) ? 8 : 2;
    std::vector<Value *> fragA;
    for (int iter = 0; iter < fragA_size; iter++) {
      Value *p = builder->CreateExtractValue(fragment_a, iter);
      fragA.push_back(p);
    }
    std::vector<Value *> fragB;
    int fragB_size = (op->type == Float(32)) ? 8 : 2;

    for (int iter = 0; iter < fragB_size; iter++) {
      Value *p = builder->CreateExtractValue(fragment_b, iter);
      fragB.push_back(p);
    }
    Value *fragment_c = codegen_buffer_pointer(
        "fragment_c" + get_expr_str(op->args[0]) + get_expr_str(op->args[1]),
        op->type, make_zero(Int(32)));
    Value *fragC[8];
    for (size_t i = 0; i < 8; i++) {

      Value *idx = ConstantInt::get(i64_t, i);
      Value *elem_ptr = builder->CreateInBoundsGEP(fragment_c, idx);
      LoadInst *lod = builder->CreateLoad(elem_ptr);
      fragC[i] = lod;
      Expr ind = make_const(Int(32), i);
      add_tbaa_metadata(lod, "fragment_c", ind);
      ;
    }
    llvm::Function *fnM;
    string mmacintr;
    if (get_expr_str(op->args[2]) == "row") {
      if (get_expr_str(op->args[3]) == "col")
        mmacintr = mma_mac_intrinsic(op->type, 16, 16, 16, row, col);
      else if (get_expr_str(op->args[3]) == "row")
        mmacintr = mma_mac_intrinsic(op->type, 16, 16, 16, row, row);
    } else if (get_expr_str(op->args[2]) == "col") {
      if (get_expr_str(op->args[3]) == "col")
        mmacintr = mma_mac_intrinsic(op->type, 16, 16, 16, col, col);
      else if (get_expr_str(op->args[3]) == "row")
        mmacintr = mma_mac_intrinsic(op->type, 16, 16, 16, col, row);
    }

    fnM = module->getFunction(mmacintr);
    internal_assert(fnM) << "Could not find MMA intrinsic "
                         << "\n";
    std::string type_strM;
    llvm::raw_string_ostream rsM(type_strM);
    std::vector<Value *> fnM_args;
    for (const auto &fa : fragA)
      fnM_args.push_back(fa);
    for (const auto &fb : fragB)
      fnM_args.push_back(fb);
    for (const auto &fc : fragC)
      fnM_args.push_back(fc);
    Value *new_mac = builder->CreateCall(fnM, fnM_args);
    for (size_t i = 0; i < 8; i++) {
      Value *p = builder->CreateExtractValue(new_mac, i);
      Value *idx = ConstantInt::get(i64_t, i);
      Value *elem_ptr = builder->CreateInBoundsGEP(fragment_c, idx);
      StoreInst *macb = builder->CreateStore(p, elem_ptr);
      Expr ind = make_const(Int(32), i);
      add_tbaa_metadata(macb, "fragment_c", ind);
      ;
    }

    value = ConstantInt::get(i32_t, 0);
  } else if (op->is_intrinsic(Call::mma_store)) {
    Value *fragment_c = codegen_buffer_pointer(
        "fragment_c" + get_expr_str(op->args[4]) + get_expr_str(op->args[5]),
        op->type, make_zero(Int(32)));
    Value *fragC[8];
    for (size_t i = 0; i < 8; i++) {
      Value *idx = ConstantInt::get(i64_t, i);
      Value *elem_ptr = builder->CreateInBoundsGEP(fragment_c, idx);
      LoadInst *lod = builder->CreateLoad(elem_ptr);
      fragC[i] = lod;
      Expr ind = make_const(Int(32), i);
      add_tbaa_metadata(lod, "fragment_c", ind);
      ;
    }
    std::string fname = get_expr_str(op->args[1]);
    Value *idx = codegen(op->args[2]);
    Type l_type = (op->type == Float(32)) ? Float(32) : Int(32);
    Value *ptr = codegen_fragment_pointer(fname, l_type, op->type, idx, 8);
    std::string storeDintr = mma_store_intrinsic(op->type, 16, 16, 16, row);
    Value *stride = codegen(op->args[3]);
    llvm::Function *fnS = module->getFunction(storeDintr);
    internal_assert(fnS) << "Could not find Store intrinsic "
                         << "\n";
    builder->CreateCall(fnS, {ptr, fragC[0], fragC[1], fragC[2], fragC[3],
                              fragC[4], fragC[5], fragC[6], fragC[7], stride});
    value = ConstantInt::get(i32_t, 0);

  }

  else if (op->is_intrinsic(Call::gpu_thread_barrier)) {
    llvm::Function *barrier0 = module->getFunction("llvm.nvvm.barrier0");
    internal_assert(barrier0)
        << "Could not find PTX barrier intrinsic (llvm.nvvm.barrier0)\n";
    builder->CreateCall(barrier0);
    value = ConstantInt::get(i32_t, 0);
  } else {
    CodeGen_LLVM::visit(op);
  }
}

string CodeGen_PTX_Dev::simt_intrinsic(const string &name) {
  if (ends_with(name, ".__thread_id_x")) {
    return "llvm.nvvm.read.ptx.sreg.tid.x";
  } else if (ends_with(name, ".__thread_id_y")) {
    return "llvm.nvvm.read.ptx.sreg.tid.y";
  } else if (ends_with(name, ".__thread_id_z")) {
    return "llvm.nvvm.read.ptx.sreg.tid.z";
  } else if (ends_with(name, ".__thread_id_w")) {
    return "llvm.nvvm.read.ptx.sreg.tid.w";
  } else if (ends_with(name, ".__block_id_x")) {
    return "llvm.nvvm.read.ptx.sreg.ctaid.x";
  } else if (ends_with(name, ".__block_id_y")) {
    return "llvm.nvvm.read.ptx.sreg.ctaid.y";
  } else if (ends_with(name, ".__block_id_z")) {
    return "llvm.nvvm.read.ptx.sreg.ctaid.z";
  } else if (ends_with(name, ".__block_id_w")) {
    return "llvm.nvvm.read.ptx.sreg.ctaid.w";
  }
  internal_error << "simt_intrinsic called on bad variable name\n";
  return "";
}

// todo: make this function a string creation for all possible types (less safe
// perhaps?)
string CodeGen_PTX_Dev::mma_load_intrinsic(const Type &type, const int m,
                                           const int n, const int k,
                                           const mma_frag &frag,
                                           const mma_frag_layout &rc) {
  if (frag == a) {
    if (m == 16 && n == 16 && k == 16) {
      if (type == Float(16)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p0i32";

      } else if (type == UInt(8)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.a.row.stride.u8.p0i32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.a.col.stride.u8.p0i32";
      }
    }

  } else if (frag == b) {
    if (m == 16 && n == 16 && k == 16) {
      if (type == Float(16)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0i32";

      } else if (type == UInt(8)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.b.row.stride.u8.p0i32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.b.col.stride.u8.p0i32";
      }
    }
  } else if (frag == c) {
    if (m == 16 && n == 16 && k == 16) {
      if (type == Float(32)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0f32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p0i32";

      } else if (type == Int(32)) {
        if (rc == row)
          return "llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p0i32";
        else if (rc == col)
          return "llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p0i32";
      }
    }
  }
  internal_error << "Undefined mma load operation\n";
  return "";
}

string CodeGen_PTX_Dev::mma_mac_intrinsic(const Type &type, const int m,
                                          const int n, const int k,
                                          const mma_frag_layout &a,
                                          const mma_frag_layout &b) {
  if (m == 16 && n == 16 && k == 16) {
    if (type == Float(32)) {
      if (a == row) {
        if (b == col)
          return "llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32";
        else if (b == row)
          return "llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32";
      } else if (a == col) {
        if (b == col)
          return "llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32";
        else if (b == row)
          return "llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32";
      }

    } else if (type == Int(32)) {
      if (a == row) {
        if (b == col)
          return "llvm.nvvm.wmma.m16n16k16.mma.row.col.u8";
        else if (b == row)
          return "llvm.nvvm.wmma.m16n16k16.mma.row.row.u8";
      } else if (a == col) {
        if (b == col)
          return "llvm.nvvm.wmma.m16n16k16.mma.col.col.u8";
        else if (b == row)
          return "llvm.nvvm.wmma.m16n16k16.mma.col.row.u8";
      }
    }
  }
  internal_error << "Undefined mma mac operation\n";
  return "";
}

string CodeGen_PTX_Dev::mma_store_intrinsic(const Type &type, const int m,
                                            const int n, const int k,
                                            const mma_frag_layout &rc) {
  if (m == 16 && n == 16 && k == 16) {
    if (type == Float(32)) {
      if (rc == row)
        return "llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32";
      else if (rc == col)
        return "llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p0i32";

    } else if (type == Int(32)) {
      if (rc == row)
        return "llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p0i32";
      else if (rc == col)
        return "llvm.nvvm.wmma.m16n16k16.store.d.col.stride.s32.p0i32";
    }
  }
  internal_error << "Undefined mma store operation\n";
  return "";
}
void CodeGen_PTX_Dev::visit(const For *loop) {
  if (is_gpu_var(loop->name)) {
    Expr simt_idx = Call::make(Int(32), simt_intrinsic(loop->name),
                               std::vector<Expr>(), Call::Extern);
    internal_assert(is_zero(loop->min));
    sym_push(loop->name, codegen(simt_idx));
    codegen(loop->body);
    sym_pop(loop->name);
  } else {
    CodeGen_LLVM::visit(loop);
  }
}

void CodeGen_PTX_Dev::visit(const Allocate *alloc) {
  user_assert(!alloc->new_expr.defined())
      << "Allocate node inside PTX kernel has custom new expression.\n"
      << "(Memoization is not supported inside GPU kernels at present.)\n";
  if (alloc->name == "__shared") {
    // PTX uses zero in address space 3 as the base address for shared memory
    Value *shared_base = Constant::getNullValue(PointerType::get(i8_t, 3));
    sym_push(alloc->name, shared_base);
  } else {
    debug(2) << "Allocate " << alloc->name << " on device\n";

    string allocation_name = alloc->name;
    debug(3) << "Pushing allocation called " << allocation_name
             << " onto the symbol table\n";

    // Jump back to the entry and generate an alloca. Note that by
    // jumping back we're rendering any expression we carry back
    // meaningless, so we had better only be dealing with
    // constants here.
    int32_t size = alloc->constant_allocation_size();
    user_assert(size > 0)
        << "Allocation " << alloc->name << " has a dynamic size. "
        << "Only fixed-size allocations are supported on the gpu. "
        << "Try storing into shared memory instead.";

    BasicBlock *here = builder->GetInsertBlock();

    builder->SetInsertPoint(entry_block);
    Value *ptr = builder->CreateAlloca(llvm_type_of(alloc->type),
                                       ConstantInt::get(i32_t, size));
    builder->SetInsertPoint(here);
    sym_push(allocation_name, ptr);
  }
  codegen(alloc->body);
}

void CodeGen_PTX_Dev::visit(const Free *f) { sym_pop(f->name); }

void CodeGen_PTX_Dev::visit(const AssertStmt *op) {
  // Discard the error message for now.
  Expr trap = Call::make(Int(32), "halide_ptx_trap", {}, Call::Extern);
  codegen(IfThenElse::make(!op->condition, Evaluate::make(trap)));
}

void CodeGen_PTX_Dev::visit(const Load *op) {
  // Do aligned 4-wide 32-bit loads as a single i128 load.
  const Ramp *r = op->index.as<Ramp>();
  // TODO: lanes >= 4, not lanes == 4
  if (is_one(op->predicate) && r && is_one(r->stride) && r->lanes == 4 &&
      op->type.bits() == 32) {
    ModulusRemainder align = op->alignment;
    if (align.modulus % 4 == 0 && align.remainder % 4 == 0) {
      Expr index = simplify(r->base / 4);
      Expr equiv = Load::make(UInt(128), op->name, index, op->image, op->param,
                              const_true(), align / 4);
      equiv = reinterpret(op->type, equiv);
      codegen(equiv);
      return;
    }
  }

  CodeGen_LLVM::visit(op);
  return;
  //}
}

void CodeGen_PTX_Dev::create_alloca(std::map<std::string, llvm::Type *> allocs,
                                    bool clear) {

  // Jump back to the entry and generate an alloca. Note that by
  // jumping back we're rendering any expression we carry back
  // meaningless, so we had better only be dealing with
  // constants here.
  if (clear) {
    for (const auto &qq : allocs)
      sym_pop(qq.first);
  }
  int32_t size = 64;

  BasicBlock *here = builder->GetInsertBlock();
  for (const auto &qq : allocs) {
    builder->SetInsertPoint(entry_block);
    std::string allocation_name = qq.first;
    debug(2) << "Allocate " << allocation_name << " on device\n";

    llvm::Type *type = qq.second;
    AllocaInst *ptr =
        builder->CreateAlloca(type, ConstantInt::get(i32_t, size));
    ptr->setAlignment(8);

    Value *pointer = ptr;
    builder->SetInsertPoint(here);
    sym_push(allocation_name, pointer);
  }

  return;
}

Value *CodeGen_PTX_Dev::codegen_fragment_pointer(std::string buffer,
                                                 Halide::Type type,
                                                 Halide::Type elem_type,
                                                 Value *indexv,
                                                 int fragment_size) {
  llvm::DataLayout d(module.get());
  Value *base_address = sym_get(buffer);
  type = upgrade_type_for_storage(type);
  llvm::Type *base_address_type = base_address->getType();
  unsigned address_space = base_address_type->getPointerAddressSpace();
  llvm::Type *load_type = llvm_type_of(type)->getPointerTo(address_space);
  llvm::Type *frag_type = llvm_type_of(elem_type)->getPointerTo(address_space);
  // If the type doesn't match the expected type, we need to pointer cast
  if ((base_address_type != frag_type) ||
      ((base_address_type->getPointerAddressSpace()) != address_space)) {
    base_address = builder->CreatePointerCast(base_address, frag_type);
  }
  if (d.getPointerSize() == 8) {
    indexv = builder->CreateIntCast(indexv, i64_t, true);
  }
  base_address = builder->CreateInBoundsGEP(base_address, indexv);
  if (load_type != base_address_type) {
    base_address = builder->CreatePointerCast(base_address, load_type);
  }

  return base_address;
}

Value *CodeGen_PTX_Dev::upgrade_for_memcpy(Value *prev_base,
                                           llvm::Type *load_type,
                                           Value *indexv) {
  llvm::DataLayout d(module.get());
  llvm::Value *base_address = prev_base;
  llvm::Type *base_address_type = base_address->getType();
  unsigned address_space = base_address_type->getPointerAddressSpace();
  llvm::Type *l_type = load_type->getPointerTo(address_space);

  // If the type doesn't match the expected type, we need to pointer cast
  if ((base_address_type != l_type)) {
    base_address = builder->CreatePointerCast(base_address, l_type);
  }

  // Promote index to 64-bit on targets that use 64-bit pointers.
  if (d.getPointerSize() == 8) {
    indexv = builder->CreateIntCast(indexv, i64_t, true);
  }
  base_address = builder->CreateInBoundsGEP(base_address, indexv);
  return base_address;
}

void CodeGen_PTX_Dev::visit(const Store *op) {

  // Issue atomic store if we are inside an Atomic node.
  if (emit_atomic_stores) {
    user_assert(is_one(op->predicate))
        << "Atomic update does not support predicated store.\n";
    user_assert(op->value.type().bits() >= 32)
        << "CUDA: 8-bit or 16-bit atomics are not supported.\n";
#if LLVM_VERSION < 90
    user_assert(op->value.type().is_scalar())
        << "CUDA atomic update does not support vectorization with LLVM "
           "version < 9.\n";
    // Generate nvvm intrinsics for the atomics if this is a float atomicAdd.
    // Otherwise defer to the llvm codegen. For llvm version >= 90, atomicrmw
    // support floats so we can also refer to llvm. Half atomics are supported
    // by compute capability 7.x or higher.
    if (op->value.type().is_float() &&
        (op->value.type().bits() == 32 ||
         (op->value.type().bits() == 64 &&
          target.has_feature(Target::CUDACapability61)))) {
      Expr val_expr = op->value;
      Expr equiv_load =
          Load::make(op->value.type(), op->name, op->index, Buffer<>(),
                     op->param, op->predicate, op->alignment);
      Expr delta =
          simplify(common_subexpression_elimination(op->value - equiv_load));
      // For atomicAdd, we check if op->value - store[index] is independent of
      // store.
      bool is_atomic_add = !expr_uses_var(delta, op->name);
      if (is_atomic_add) {
        Value *ptr =
            codegen_buffer_pointer(op->name, op->value.type(), op->index);
        Value *val = codegen(delta);
        llvm::Function *intrin = nullptr;
        if (op->value.type().bits() == 32) {
          intrin = module->getFunction("llvm.nvvm.atomic.load.add.f32.p0f32");
          internal_assert(intrin) << "Could not find atomic intrinsics "
                                     "llvm.nvvm.atomic.load.add.f32.p0f32\n";
        } else {
          internal_assert(op->value.type().bits() == 64);
          intrin = module->getFunction("llvm.nvvm.atomic.load.add.f64.p0f64");
          internal_assert(intrin) << "Could not find atomic intrinsics "
                                     "llvm.nvvm.atomic.load.add.f64.p0f64\n";
        }
        value = builder->CreateCall(intrin, {ptr, val});
        return;
      }
    }
#endif
  }

  // Do aligned 4-wide 32-bit stores as a single i128 store.
  const Ramp *r = op->index.as<Ramp>();
  // TODO: lanes >= 4, not lanes == 4
  if (is_one(op->predicate) && r && is_one(r->stride) && r->lanes == 4 &&
      op->value.type().bits() == 32) {
    ModulusRemainder align = op->alignment;
    if (align.modulus % 4 == 0 && align.remainder % 4 == 0) {
      Expr index = simplify(r->base / 4);
      Expr value = reinterpret(UInt(128), op->value);
      Stmt equiv = Store::make(op->name, value, index, op->param, const_true(),
                               align / 4);
      codegen(equiv);
      return;
    }
  }

  CodeGen_LLVM::visit(op);
}

void CodeGen_PTX_Dev::visit(const Atomic *op) {
  // CUDA requires all the threads in a warp to perform the same operations,
  // which means our mutex will lead to deadlock.
  user_assert(op->mutex_name.empty())
      << "The atomic update requires a mutex lock, which is not supported in "
         "CUDA.\n";

  // Issue atomic stores.
  ScopedValue<bool> old_emit_atomic_stores(emit_atomic_stores, true);
  IRVisitor::visit(op);
}

void CodeGen_PTX_Dev::visit(const TensorOp *op) {

  ScopedValue<std::vector<Expr>> top_arguments(top_args, op->args);
  std::map<std::string, llvm::Type *> allocs;
  Expr ld = op->args[2];
  Expr WMMA_M = (make_const(Int(32), 16));
  Expr blx = (Call::make(Int(32), "llvm.nvvm.read.ptx.sreg.ctaid.x",
                         std::vector<Expr>(), Call::Extern));
  Expr WARPS_PER_BLOCK = make_const(Int(32), 8);
  Expr BLOCK_ROW_WARPS = make_const(Int(32), 2);
  Expr BLOCK_COL_WARPS = make_const(Int(32), 4);
  Expr WARP_ROW_TILES = make_const(Int(32), 4);
  Expr WARP_COL_TILES = make_const(Int(32), 2);
  Expr BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
  Expr BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;

  Expr GDIM = make_const(Int(32), 68);
  Expr N_TILES = (ld / WMMA_M);

  llvm::Type *aggregate_cd = nullptr;
  aggregate_cd = llvm_type_of(op->types[2]);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      std::string fname = "fragment_c(0 + ";
      fname.append(std::to_string(i));
      fname.append(")");
      fname.append("(0 + ");
      fname.append(std::to_string(j));
      fname.append(")");
      allocs.emplace(fname, aggregate_cd);
    }
  }

  Value *shared_base = Constant::getNullValue(PointerType::get(i8_t, 3));
  sym_push("__shared", shared_base);
  create_alloca(allocs, false);

  // lower the body!
  Stmt loop_body = common_subexpression_elimination(lower_tensor_body(op));
  std::cout << "Before Unroll\n" << loop_body << "\n";
  loop_body = (common_subexpression_elimination(unroll_loops(loop_body)));
  codegen(loop_body);
}

string CodeGen_PTX_Dev::march() const { return "nvptx64"; }

string CodeGen_PTX_Dev::mcpu() const {
  if (target.has_feature(Target::CUDACapability61)) {
    return "sm_75";
  } else if (target.has_feature(Target::CUDACapability50)) {
    return "sm_70";
  } else if (target.has_feature(Target::CUDACapability35)) {
    return "sm_70";
  } else if (target.has_feature(Target::CUDACapability32)) {
    return "sm_70";
  } else if (target.has_feature(Target::CUDACapability30)) {
    return "sm_70";
  } else {
    return "sm_70";
  }
}

string CodeGen_PTX_Dev::mattrs() const {
  if (target.has_feature(Target::CUDACapability61)) {
    return "+ptx60";
  } else if (target.features_any_of(
                 {Target::CUDACapability32, Target::CUDACapability50})) {
    // Need ptx isa 4.0.
    return "+ptx60";
  } else {
    // Use the default. For llvm 3.5 it's ptx 3.2.
    return "+ptx60";
  }
}

bool CodeGen_PTX_Dev::use_soft_float_abi() const { return false; }

vector<char> CodeGen_PTX_Dev::compile_to_src() {

#ifdef WITH_PTX

  debug(2) << "In CodeGen_PTX_Dev::compile_to_src";

  // DISABLED - hooked in here to force PrintBeforeAll option - seems to be the
  // only way?
  /*char* argv[] = { "llc", "-print-before-all" };*/
  /*int argc = sizeof(argv)/sizeof(char*);*/
  /*cl::ParseCommandLineOptions(argc, argv, "Halide PTX internal compiler\n");*/

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  internal_assert(target) << err_str << "\n";

  TargetOptions options;
  options.PrintMachineCode = false;
  options.AllowFPOpFusion = FPOpFusion::Fast;
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;
  options.StackAlignmentOverride = 0;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), mcpu(), mattrs(), options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small, CodeGenOpt::Aggressive));

  internal_assert(target_machine.get()) << "Could not allocate target machine!";

  module->setDataLayout(target_machine->createDataLayout());
  debug(1) << "Target triple of nvvm module: " << module->getTargetTriple()
           << "\n";
  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  legacy::FunctionPassManager function_pass_manager(module.get());
  legacy::PassManager module_pass_manager;

  module_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));
  function_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const int kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  PassManagerBuilder b;
  b.OptLevel = 3;
  b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
  b.LoopVectorize = true;
  b.SLPVectorize = true;

  target_machine->adjustPassManager(b);

  b.populateFunctionPassManager(function_pass_manager);
  b.populateModulePassManager(module_pass_manager);

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  // Output string stream

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, TargetMachine::CGFT_AssemblyFile,
      true);
  if (fail) {
    internal_error << "Failed to set up passes to emit PTX source\n";
  }

  // Run optimization passes
  function_pass_manager.doInitialization();
  for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
    function_pass_manager.run(*i);
  }
  function_pass_manager.doFinalization();
  module_pass_manager.run(*module);

  if (debug::debug_level() >= 2) {
    dump();
  }
  debug(2) << "Done with CodeGen_PTX_Dev::compile_to_src";

  debug(1) << "PTX kernel:\n" << outstr.c_str() << "\n";

  vector<char> buffer(outstr.begin(), outstr.end());

  // Dump the SASS too if the cuda SDK is in the path
  if (debug::debug_level() >= 2) {
    debug(2) << "Compiling PTX to SASS. Will fail if CUDA SDK is not installed "
                "(and in the path).\n";

    TemporaryFile ptx(get_current_kernel_name(), ".ptx");
    TemporaryFile sass(get_current_kernel_name(), ".sass");

    std::ofstream f(ptx.pathname());
    f.write(buffer.data(), buffer.size());
    f.close();

    string cmd = "ptxas --gpu-name " + mcpu() + " " + ptx.pathname() + " -o " +
                 sass.pathname();
    if (system(cmd.c_str()) == 0) {
      cmd = "nvdisasm " + sass.pathname();
      int ret = system(cmd.c_str());
      (void)ret; // Don't care if it fails
    }

    // Note: It works to embed the contents of the .sass file in
    // the buffer instead of the ptx source, and this could help
    // with app startup times. Expose via the target?
    /*
    {
        std::ifstream f(sass.pathname());
        buffer.clear();
        f.seekg(0, std::ios_base::end);
        std::streampos sz = f.tellg();
        buffer.resize(sz);
        f.seekg(0, std::ios_base::beg);
        f.read(buffer.data(), sz);
    }
    */
  }

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
#else // WITH_PTX
  return vector<char>();
#endif
}

int CodeGen_PTX_Dev::native_vector_bits() const {
  // PTX doesn't really do vectorization. The widest type is a double.
  return 64;
}

string CodeGen_PTX_Dev::get_current_kernel_name() {
  return function->getName();
}

void CodeGen_PTX_Dev::dump() { module->print(dbgs(), nullptr, false, true); }

std::string CodeGen_PTX_Dev::print_gpu_name(const std::string &name) {
  return name;
}

bool CodeGen_PTX_Dev::supports_atomic_add(const Type &t) const {
  if (t.bits() < 32) {
    // TODO: Half atomics are supported by compute capability 7.x or higher.
    return false;
  }
  if (t.is_int_or_uint()) {
    return true;
  }
  if (t.is_float() && t.bits() == 32) {
    return true;
  }
  if (t.is_float() && t.bits() == 64) {
    // double atomics are supported since CC6.1
    return target.has_feature(Target::CUDACapability61);
  }
  return false;
}

} // namespace Internal
} // namespace Halide
