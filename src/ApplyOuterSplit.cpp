#include "ApplyOuterSplit.h"
#include "ApplySplit.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::map;
using std::string;
using std::vector;

vector<ApplySplitResult>
apply_outer_split(Split &split, bool is_update, string prefix,
                  map<string, Expr> &dim_extent_alignment) {
  vector<ApplySplitResult> result;

  Expr outer = Variable::make(Int(32), prefix + split.outer);
  Expr outer_max = Variable::make(Int(32), prefix + split.outer + ".loop_max");
  // Expr inner = Variable::make(Int(32), prefix + split.inner);
  Expr old_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
  Expr old_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");
  Expr old_extent =
      Variable::make(Int(32), prefix + split.old_var + ".loop_extent");

  dim_extent_alignment[split.inner] = split.factor;

  Expr base = outer * split.factor + old_min;
  string base_name = prefix + split.inner + ".base";
  Expr base_var = Variable::make(Int(32), base_name);
  string old_var_name = prefix + split.old_var;
  Expr old_var = Variable::make(Int(32), old_var_name);

  map<string, Expr>::iterator iter = dim_extent_alignment.find(split.old_var);

  TailStrategy tail = split.tail;
  internal_assert(tail != TailStrategy::Auto)
      << "An explicit tail strategy should exist at this point\n";

  if ((iter != dim_extent_alignment.end()) &&
      is_zero(simplify(iter->second % split.factor))) {
    // We have proved that the split factor divides the
    // old extent. No need to adjust the base or add an if
    // statement.
    dim_extent_alignment[split.outer] = iter->second / split.factor;
  } else if (is_negative_const(split.factor) || is_zero(split.factor)) {
    user_error << "Can't split " << split.old_var << " by " << split.factor
               << ". Split factors must be strictly positive\n";
  } else if (is_one(split.factor)) {
    // The split factor trivially divides the old extent,
    // but we know nothing new about the outer dimension.
  } else if (tail == TailStrategy::GuardWithIf) {
    // It's an exact split but we failed to prove that the
    // extent divides the factor. Use predication.

    // Make a var representing the original var minus its
    // min. It's important that this is a single Var so
    // that bounds inference has a chance of understanding
    // what it means for it to be limited by the if
    // statement's condition.
    Expr rebased = outer * split.factor;
    string rebased_var_name = prefix + split.old_var + ".rebased";
    Expr rebased_var = Variable::make(Int(32), rebased_var_name);

    result.push_back(ApplySplitResult(prefix + split.old_var,
                                      rebased_var + old_min,
                                      ApplySplitResult::Substitution));

    // Tell Halide to optimize for the case in which this
    // condition is true by partitioning some outer loop.
    Expr cond = likely(rebased_var < old_extent);
    result.push_back(ApplySplitResult(cond));
    result.push_back(
        ApplySplitResult(rebased_var_name, rebased, ApplySplitResult::LetStmt));

  } else if (tail == TailStrategy::ShiftInwards) {
    // Adjust the base downwards to not compute off the
    // end of the realization.

    // We'll only mark the base as likely (triggering a loop
    // partition) if we're at or inside the innermost
    // non-trivial loop.
    base = likely_if_innermost(base);

    base = Min::make(base, old_max + (1 - split.factor));
  } else {
    internal_assert(tail == TailStrategy::RoundUp);
  }

  // Substitute in the new expression for the split variable ...
  result.push_back(
      ApplySplitResult(old_var_name, base_var, ApplySplitResult::Substitution));
  // ... but also define it as a let for the benefit of bounds inference.
  result.push_back(
      ApplySplitResult(old_var_name, base_var, ApplySplitResult::LetStmt));
  result.push_back(
      ApplySplitResult(base_name, base, ApplySplitResult::LetStmt));

  return result;
}

vector<std::pair<string, Expr>>
compute_loop_bounds_after_outer_split(Split &split, string prefix) {
  // Define the bounds on the split dimensions using the bounds
  // on the function args. If it is a purify, we should use the bounds
  // from the dims instead.

  vector<std::pair<string, Expr>> let_stmts;

  Expr old_var_extent =
      Variable::make(Int(32), prefix + split.old_var + ".loop_extent");
  Expr old_var_max =
      Variable::make(Int(32), prefix + split.old_var + ".loop_max");
  Expr old_var_min =
      Variable::make(Int(32), prefix + split.old_var + ".loop_min");

  //        Expr inner_extent = split.factor;
  Expr outer_extent = (old_var_max - old_var_min + split.factor) / split.factor;
  let_stmts.push_back({prefix + split.outer + ".loop_min", 0});
  let_stmts.push_back({prefix + split.outer + ".loop_max", outer_extent - 1});
  let_stmts.push_back({prefix + split.outer + ".loop_extent", outer_extent});

  // Do nothing for purify

  return let_stmts;
}

} // namespace Internal
} // namespace Halide
