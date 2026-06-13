#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "search_trace.h"
#include "difficulty.h"
#include "percentile.h"
#include "difficulty_matrix.h"

namespace lidef {

// Result of an EF prediction, with diagnostic info.
struct EfPrediction {
    int ef;             // predicted search budget
    int B_M;            // macro difficulty bucket used
    int B_m;            // micro difficulty bucket used
    bool out_of_range;  // true if even the largest scheduled ef
                         // did not reach R_t for this cell
};

class LidEfPredictor {
public:
    LidEfPredictor(PercentileNormalizer macro_norm,
                    PercentileNormalizer micro_norm,
                    std::vector<std::vector<Cell>> table,
                    std::vector<int> ef_schedule)
        : macro_norm_(std::move(macro_norm)),
          micro_norm_(std::move(micro_norm)),
          table_(std::move(table)),
          ef_schedule_(std::move(ef_schedule)) {}

    // Given a probe-search trace and a target recall, predict the
    // search budget for the full search.
    EfPrediction predict(const SearchTrace& probe_trace, double R_t) const {
        DifficultyScores scores = compute_difficulty(probe_trace);
        int B_M = macro_norm_.bucket(scores.M);
        int B_m = micro_norm_.bucket(scores.m);

        const Cell& cell = table_[B_M][B_m];
        EfPrediction out;
        out.B_M = B_M;
        out.B_m = B_m;
        out.ef = interpolate_ef(cell, R_t, out.out_of_range);
        return out;
    }

private:
    PercentileNormalizer macro_norm_;
    PercentileNormalizer micro_norm_;
    std::vector<std::vector<Cell>> table_;
    std::vector<int> ef_schedule_;

    // Curve is assumed non-decreasing in recall (guaranteed by
    // DifficultyMatrixBuilder::finalize's pass (a)). Finds smallest
    // ef such that recall(ef) >= R_t, interpolating linearly between
    // grid points when necessary.
    static int interpolate_ef(const Cell& cell, double R_t, bool& out_of_range) {
        out_of_range = false;
        const auto& curve = cell.curve;
        if (curve.empty()) {
            out_of_range = true;
            return 0;
        }

        // If even the smallest ef already meets R_t, use it.
        if (curve.front().recall >= R_t) {
            return curve.front().ef;
        }

        for (size_t k = 1; k < curve.size(); ++k) {
            if (curve[k].recall >= R_t) {
                const auto& lo = curve[k - 1];
                const auto& hi = curve[k];
                if (hi.recall == lo.recall) {
                    // degenerate: flat segment, return upper endpoint
                    return hi.ef;
                }
                double frac = (R_t - lo.recall) / (hi.recall - lo.recall);
                double ef_interp = lo.ef + frac * (hi.ef - lo.ef);
                return static_cast<int>(std::ceil(ef_interp));
            }
        }

        // No budget in the schedule reaches R_t for this cell.
        out_of_range = true;
        return curve.back().ef;
    }
};

} // namespace lidef
