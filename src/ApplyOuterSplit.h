#ifndef APPLY_SPLIT_H
#define APPLY_SPLIT_H

/** \file
 *
 * Defines method that returns a list of let stmts, substitutions, and
 * predicates to be added given a split schedule.
 */

#include <map>
#include <utility>
#include <vector>

#include "ApplySplit.h"
#include "IR.h"
#include "Schedule.h"
namespace Halide {
namespace Internal {
std::vector<ApplySplitResult>
apply_outer_split(Split split, bool is_update, std::string prefix,
                  std::map<std::string, Expr> &dim_extent_alignment);

std::vector<std::pair<std::string, Expr>>
compute_loop_bounds_after_outer_split(Split split, std::string prefix);

} // namespace Internal
} // namespace Halide

#endif
