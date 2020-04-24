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

Stmt make_equiv_body(const std::vector<Expr> &targs,
                     const std::vector<Type> &types) {

  // initialize constants
  Expr ld = targs[2];
  Expr WMMA_M = (make_const(Int(32), 16));

  Expr blx = targs[4]; //(Call::make(Int(32), "llvm.nvvm.read.ptx.sreg.ctaid.x",
                       // std::vector<Expr>(), Call::Extern));;
  Expr thx = targs[3]; //(Call::make(Int(32), "llvm.nvvm.read.ptx.sreg.tid.x",
                       // std::vector<Expr>(), Call::Extern));;
  Expr WARP_SIZE = (make_const(Int(32), 32));
  Expr WARPS_PER_BLOCK = make_const(Int(32), 8);
  Expr THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
  Expr BLOCK_ROW_WARPS = make_const(Int(32), 2);
  Expr BLOCK_COL_WARPS = make_const(Int(32), 4);
  Expr WARP_ROW_TILES = make_const(Int(32), 4);
  Expr WARP_COL_TILES = make_const(Int(32), 2);
  Expr BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
  Expr BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
  Expr GLOBAL_MEM_STRIDE = ld;
  Expr SHMEM_STRIDE = make_const(Int(32), 128);
  ;
  Expr SHMEM_OFFSET = make_const(Int(32), 64);
  ;
  Expr SKEW_FACTOR = (types[2] == Float(32)) ? 8 : 16;
  Expr GDIM = make_const(Int(32), 68);
  Expr shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
  ;
  Expr warpID = (thx / 32);
  Expr laneID = (thx % 32);
  Expr CHUNK_COPY_LINES_PER_WARP = make_const(Int(32), 4);
  ;
  ;
  Expr CHUNK_COPY_LINE_LANES = make_const(Int(32), 8);
  ;
  ;
  Expr shmem_warp_tile_ptr =
      (warpID / 2) * SHMEM_STRIDE * WMMA_M * 2 + (warpID % 2) * SHMEM_OFFSET;
  Expr shmem_warp_stream_ptr = warpID * SHMEM_STRIDE * WMMA_M;
  Expr N_TILES = (ld / WMMA_M);
  Expr block_pos = Variable::make(Int(32), "block_pos");
  // block_pos

  Expr block_tile_i =
      (cast<unsigned int>(block_pos * BLOCK_ROW_TILES) / N_TILES) *
      (BLOCK_COL_TILES);
  Expr block_cond = (block_tile_i >= N_TILES);
  Expr block_step = GDIM;
  Expr block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
  Expr gmem_idx = (block_tile_i + warpID) * WMMA_M * ld + block_tile_j * WMMA_M;
  Expr input_a = targs[7];
  Expr input_b = targs[8];
  Expr output = targs[6];
  Expr CHUNK_K = (types[2] == Float(32)) ? 4 : 8;
  // copy_C_to_shmem
  Expr C_to_shmem = Variable::make(Int(32), "C0");
  Stmt copy_C_to_Shared_mem;
  Expr Store_index_Cg =
      simplify(shmem_warp_stream_ptr + SHMEM_STRIDE * C_to_shmem);
  Expr Load_index_Cg = simplify(gmem_idx + GLOBAL_MEM_STRIDE * C_to_shmem);
  vector<Expr> args_to_C_shmem;

  args_to_C_shmem.push_back(Load_index_Cg);
  args_to_C_shmem.push_back(Store_index_Cg);
  args_to_C_shmem.push_back(laneID);
  args_to_C_shmem.push_back(output);
  Stmt body_copy_C_to_Shared_mem =
      Evaluate::make(Call::make(types[2], Call::mma_memcpy_C_to_shared,
                                args_to_C_shmem, Call::Intrinsic));
  // variable for this;

  copy_C_to_Shared_mem =
      For::make("C0", make_zero(Int(32)), WMMA_M, ForType::Unrolled,
                DeviceAPI::None, body_copy_C_to_Shared_mem);

  // barriers
  Stmt barrier_call = Evaluate::make(Call::make(
      Int(32), Call::gpu_thread_barrier, vector<Expr>(), Call::Intrinsic));

  // now load C frags

  Expr Cli = Variable::make(Int(32), "Cli");
  Expr Clj = Variable::make(Int(32), "Clj");
  Expr tptr = shmem_warp_tile_ptr + Cli * SHMEM_STRIDE * WMMA_M + Clj * WMMA_M;
  vector<Expr> args_lc;
  args_lc.push_back(make_const(Int(32), 2));
  args_lc.push_back(Expr("__shared"));
  args_lc.push_back(tptr);
  args_lc.push_back(SHMEM_STRIDE);
  args_lc.push_back(Cli);
  args_lc.push_back(Clj);
  Stmt LoadC = Evaluate::make(
      Call::make(types[2], Call::mma_load, args_lc, Call::Intrinsic));

  // make the body j loop
  Stmt LoadC_inner = For::make("Clj", make_zero(Int(32)), WARP_ROW_TILES,
                               ForType::Unrolled, DeviceAPI::None, LoadC);
  // make the i loop
  Stmt LoadC_outer = For::make("Cli", make_zero(Int(32)), WARP_COL_TILES,
                               ForType::Unrolled, DeviceAPI::None, LoadC_inner);

  // now Create the tile_k loop
  Expr tile_k = Variable::make(Int(32), "tile_k");

  // make the calls to copy A/B to shmem
  Expr ABl = Variable::make(Int(32), "ABl");
  vector<Expr> AB_args;
  Expr indxA = block_tile_i * WMMA_M * ld;
  Expr offsetA = WMMA_M * ld * (warpID % 4) * 2;
  Expr indxB = block_tile_j * WMMA_M * ld;
  Expr fa = WMMA_M * (warpID % (8 / 2)) * 2;
  Expr fb = (WMMA_M * (warpID % (8 / 2)) * 2 + 128);
  Expr cond_idx = warpID < (4);
  Expr eshmem_idx =
      Select::make(cond_idx, fa, fb) + laneID / CHUNK_COPY_LINE_LANES;
  Expr memcpy_offset = laneID % CHUNK_COPY_LINE_LANES;
  Expr elane_ptrA = (CHUNK_K * tile_k * WMMA_M +
                     cast<int>(laneID / CHUNK_COPY_LINE_LANES) * ld);
  Expr elane_ptrB = (CHUNK_K * tile_k * WMMA_M +
                     cast<int>(laneID / CHUNK_COPY_LINE_LANES) * ld);
  Expr Store_index_l =
      simplify((eshmem_idx + 4 * ABl) * (CHUNK_K * WMMA_M + SKEW_FACTOR));
  Expr offset_ptr = offsetA + elane_ptrA + 4 * ld * ABl;
  AB_args.push_back(warpID);
  AB_args.push_back(input_a);
  AB_args.push_back(input_b);
  AB_args.push_back(offset_ptr);
  AB_args.push_back(indxA);
  AB_args.push_back(indxB);
  AB_args.push_back(memcpy_offset);
  AB_args.push_back(Store_index_l);
  Stmt LoadAB_body = Evaluate::make(
      Call::make(types[0], Call::mma_memcpy_AB, AB_args, Call::Intrinsic));
  // and the loop
  Stmt LoadAB = For::make("ABl", make_zero(Int(32)), 8, ForType::Unrolled,
                          DeviceAPI::None, LoadAB_body);

  // now the k_step loops
  Expr k_step = Variable::make(Int(32), "k_step");
  // inner i loop var
  Expr k_step_i = Variable::make(Int(32), "k_step_i");
  // inner j loop var
  Expr k_step_j = Variable::make(Int(32), "k_step_j");
  // call to loadAs
  vector<Expr> args_la;
  Expr shmem_idxa = ((warpID / 2) * WMMA_M * 2 + (k_step_i * WMMA_M));
  Expr shmem_idx_a = shmem_idxa * (WMMA_M * CHUNK_K + SKEW_FACTOR);
  Expr off_A = k_step * 16;
  Expr tile_ptr = (shmem_idx_a + off_A);
  std::cout << "make Call loadA\n";
  args_la.push_back(make_const(Int(32), 0));
  args_la.push_back(Expr("__shared"));
  args_la.push_back(tile_ptr);
  Expr arg_strideA = (WMMA_M * CHUNK_K) + SKEW_FACTOR;
  args_la.push_back(arg_strideA);
  args_la.push_back(k_step_i);
  args_la.push_back(make_const(Int(32), 0));
  Stmt loada = Evaluate::make(
      Call::make(types[0], Call::mma_load, args_la, Call::Intrinsic));

  std::cout << "make Call loadB\n";
  // make call loadB
  Expr shmem_idxb =
      (shmem_idx_b_off + (WARP_ROW_TILES * WMMA_M) * (warpID % 2) +
       k_step_j * WMMA_M);
  Expr shmem_idx_b = shmem_idxb * (WMMA_M * CHUNK_K + SKEW_FACTOR);
  Expr off_B = k_step * WMMA_M;
  Expr tile_ptrb = (shmem_idx_b + off_B);
  std::vector<Expr> args_lb;
  args_lb.push_back(make_const(Int(32), 1));
  args_lb.push_back(Expr("__shared"));
  args_lb.push_back(tile_ptrb);
  Expr arg_strideB = WMMA_M * CHUNK_K + SKEW_FACTOR;
  args_lb.push_back(arg_strideB);
  args_lb.push_back(make_const(Int(32), 0));
  args_lb.push_back(k_step_j);
  Stmt loadb = Evaluate::make(
      Call::make(types[1], Call::mma_load, args_lb, Call::Intrinsic));
  std::cout << "make Call mma\n";
  // now call mma_mac
  Stmt mma = Evaluate::make(Call::make(types[2], Call::mma_operation,
                                       {k_step_i, k_step_j}, Call::Intrinsic));
  Stmt k_step_j_body = Block::make({loadb, mma});
  Stmt k_step_j_loop =
      For::make("k_step_j", make_zero(Int(32)), WARP_ROW_TILES,
                ForType::Unrolled, DeviceAPI::None, k_step_j_body);

  // k step_ i loop ;
  Stmt k_step_i_body = Block::make({loada, k_step_j_loop});
  Stmt k_step_i_loop =
      For::make("k_step_i", make_zero(Int(32)), WARP_COL_TILES,
                ForType::Unrolled, DeviceAPI::None, k_step_i_body);

  // k_step body
  // Stmt k_step_body = Block::make({k_step_i_loop});
  Stmt k_step_loop =
      For::make("k_step", make_zero(Int(32)), CHUNK_K, ForType::Unrolled,
                DeviceAPI::None, k_step_i_loop);

  // now make the tile_k loop
  Stmt tile_k_body =
      Block::make({LoadAB, barrier_call, k_step_loop, barrier_call});
  Expr tile_k_extent = N_TILES / CHUNK_K;

  Stmt tile_k_loop;
  if (can_prove(ld > 8192))
    tile_k_loop = For::make("tile_k", make_zero(Int(32)), tile_k_extent,
                            ForType::Serial, DeviceAPI::None, tile_k_body);
  else
    tile_k_loop = For::make("tile_k", make_zero(Int(32)), tile_k_extent,
                            ForType::Unrolled, DeviceAPI::None, tile_k_body);

  // now store d frags
  std::cout << "make Call Store D\n";
  Expr Sdi = Variable::make(Int(32), "Sdi");
  Expr Sdj = Variable::make(Int(32), "Sdj");
  Expr tptrd = shmem_warp_tile_ptr + Sdi * WMMA_M * SHMEM_STRIDE + Sdj * WMMA_M;
  std::vector<Expr> args_sd;
  args_sd.push_back(make_const(Int(32), 2));
  args_sd.push_back(Expr("__shared"));
  args_sd.push_back(tptrd);
  args_sd.push_back(SHMEM_STRIDE);

  args_sd.push_back(Sdi);
  args_sd.push_back(Sdj);
  Stmt StoreD = Evaluate::make(
      Call::make(types[2], Call::mma_store, args_sd, Call::Intrinsic));
  // make the body j loop
  Stmt StoreD_inner = For::make("Sdj", make_zero(Int(32)), WARP_ROW_TILES,
                                ForType::Unrolled, DeviceAPI::None, StoreD);
  // make the i loop
  Stmt StoreD_outer =
      For::make("Sdi", make_zero(Int(32)), WARP_COL_TILES, ForType::Unrolled,
                DeviceAPI::None, StoreD_inner);
  std::cout << "make Call memcpy_D\n";
  // finally store D to global mem
  Expr C_to_gmem = Variable::make(Int(32), "C1");
  Stmt copy_C_to_Global_mem;
  vector<Expr> args_to_C_gmem;
  Expr Load_index_Cs =
      simplify(shmem_warp_stream_ptr + SHMEM_STRIDE * C_to_gmem);
  Expr Store_index_Cs = simplify(gmem_idx + GLOBAL_MEM_STRIDE * C_to_gmem);
  args_to_C_gmem.push_back(Load_index_Cs);
  args_to_C_gmem.push_back(Store_index_Cs);
  args_to_C_gmem.push_back(laneID);
  args_to_C_gmem.push_back(output);

  Stmt body_copy_C_to_Global_mem = Evaluate::make(Call::make(
      types[2], Call::mma_memcpy_C_to_global, args_to_C_gmem, Call::Intrinsic));
  // variable for this;
  copy_C_to_Global_mem =
      For::make("C1", make_zero(Int(32)), WMMA_M, ForType::Unrolled,
                DeviceAPI::None, body_copy_C_to_Global_mem);

  // now make the block_pos loop
  Stmt true_block_pos_body =
      Block::make({copy_C_to_Shared_mem, barrier_call, LoadC_outer,
                   barrier_call, tile_k_loop, StoreD_outer, barrier_call,
                   copy_C_to_Global_mem, barrier_call});
  // now put it all together
  Stmt new_body =
      DoWhile::make("block_pos", blx, block_cond, block_step, ForType::Serial,
                    DeviceAPI::None, true_block_pos_body);
  return new_body;
}

Stmt lower_tensor_body(const TensorOp *op) {
  Stmt s = make_equiv_body(op->args, op->types);

  return s;
}

} // namespace Internal
} // namespace Halide
