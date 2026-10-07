// glop_presolve: run OR-Tools' glop LP presolve on an MPS file and write the
// presolved LP as a binary MPModelProto.
//
//   glop_presolve [--verbose] <input.mps> <output.pb>
//
// Integrality is relaxed, so the LP relaxation is presolved. Glop parameters
// mirror PDLP's own glop presolve (PreprocessSolver::PreprocessorParameters in
// ortools/pdlp/primal_dual_hybrid_gradient.cc): no dualization, no implied-free
// preprocessor (it relaxes variable bounds), no scaling (Jaddle scales itself).
//
// The last line printed to stdout is a JSON object with the preprocessor status
// and the objective scaling factor, which MPModelProto can't carry: the true
// objective of the presolved LP is scaling_factor * (c^T x + objective_offset).
// With --verbose, glop's presolve log precedes it on stdout (glop's logger
// only writes to stdout or LOG(INFO)).

#include <cstdio>
#include <fstream>
#include <string>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ortools/glop/parameters.pb.h"
#include "ortools/glop/preprocessor.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/lp_types.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"

namespace glop = operations_research::glop;

int main(int argc, char** argv) {
  bool verbose = false;
  std::string input, output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--verbose") {
      verbose = true;
    } else if (input.empty()) {
      input = arg;
    } else if (output.empty()) {
      output = arg;
    } else {
      input.clear();
      break;
    }
  }
  if (input.empty() || output.empty()) {
    std::fprintf(stderr, "usage: %s [--verbose] <input.mps> <output.pb>\n",
                 argv[0]);
    return 2;
  }
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kWarning);

  const absl::Time t_read = absl::Now();
  absl::StatusOr<operations_research::MPModelProto> model =
      glop::MpsFileToMPModelProto(input);
  if (!model.ok()) {
    std::fprintf(stderr, "failed to read %s: %s\n", input.c_str(),
                 std::string(model.status().message()).c_str());
    return 1;
  }
  for (auto& var : *model->mutable_variable()) var.set_is_integer(false);
  const int orig_rows = model->constraint_size();
  const int orig_cols = model->variable_size();

  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);
  model->Clear();
  const double read_seconds = absl::ToDoubleSeconds(absl::Now() - t_read);

  glop::GlopParameters params;
  params.set_solve_dual_problem(glop::GlopParameters::NEVER_DO);
  params.set_use_implied_free_preprocessor(false);
  params.set_use_scaling(false);
  params.set_log_search_progress(verbose);
  params.set_log_to_stdout(true);

  const absl::Time t_presolve = absl::Now();
  glop::MainLpPreprocessor preprocessor(&params);
  preprocessor.Run(&lp);
  const double presolve_seconds =
      absl::ToDoubleSeconds(absl::Now() - t_presolve);

  operations_research::MPModelProto presolved;
  glop::LinearProgramToMPModelProto(lp, &presolved);
  std::ofstream out(output, std::ios::binary | std::ios::trunc);
  if (!out || !presolved.SerializeToOstream(&out)) {
    std::fprintf(stderr, "failed to write %s\n", output.c_str());
    return 1;
  }

  std::printf(
      "{\"status\": \"%s\", \"objective_scaling_factor\": %.17g, "
      "\"orig_rows\": %d, \"orig_cols\": %d, \"rows\": %d, \"cols\": %d, "
      "\"read_seconds\": %.6f, \"presolve_seconds\": %.6f}\n",
      glop::GetProblemStatusString(preprocessor.status()).c_str(),
      lp.objective_scaling_factor(), orig_rows, orig_cols,
      presolved.constraint_size(), presolved.variable_size(), read_seconds,
      presolve_seconds);
  return 0;
}
