#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include "search_trace.h"

namespace lidef {

// ---------------------------------------------------------------------
// Macro difficulty: layer expansion hardness
// ---------------------------------------------------------------------

// Per-layer expansion hardness E_l(q) = log(1+n_l) / (1 + rho_l)
// where rho_l = (b_{l-1} - b_l) / (b_{l-1} + eps), rho_l in [0,1).
// trace.b_l must have length L+1 (layers 0..L), trace.n_l same length.
// Layer L (topmost) has no "previous" layer, so we treat b_{L} as its
// own b_{l-1} (rho_L = 0, i.e. E_L = log(1+n_L)).
inline std::vector<double> layer_expansion_hardness(const SearchTrace& trace,
                                                      double eps = 1e-9) {
    const size_t L1 = trace.n_l.size(); // = L+1
    if (trace.b_l.size() != L1) {
        throw std::invalid_argument("n_l and b_l must have the same length");
    }
    std::vector<double> E(L1, 0.0);

    for (size_t l = 0; l < L1; ++l) {
        double n_l = static_cast<double>(trace.n_l[l]);
        double rho_l;
        if (l + 1 == L1) {
            // topmost layer: no previous layer to compare against
            rho_l = 0.0;
        } else {
            double b_prev = trace.b_l[l + 1]; // b_{l-1} in math notation
                                               // (vector indexed 0=base..L=top,
                                               //  so "previous" = higher index)
            double b_curr = trace.b_l[l];
            rho_l = (b_prev - b_curr) / (b_prev + eps);
            // clamp to [0,1) for numerical safety
            if (rho_l < 0.0) rho_l = 0.0;
            if (rho_l >= 1.0) rho_l = 1.0 - 1e-12;
        }
        E[l] = std::log1p(n_l) / (1.0 + rho_l);
    }
    return E;
}

// Weighted sum of per-layer hardness, weights emphasizing upper layers:
// w_l = 2^l / sum_j 2^j   (l=0 base layer, l=L topmost layer)
inline double macro_difficulty(const SearchTrace& trace, double eps = 1e-9) {
    std::vector<double> E = layer_expansion_hardness(trace, eps);
    const size_t L1 = E.size();
    if (L1 == 0) return 0.0;

    // weight normalization constant: sum_{j=0}^{L} 2^j = 2^{L+1} - 1
    double denom = std::pow(2.0, static_cast<double>(L1)) - 1.0;

    double M = 0.0;
    for (size_t l = 0; l < L1; ++l) {
        double w_l = std::pow(2.0, static_cast<double>(l)) / denom;
        M += w_l * E[l];
    }
    return M;
}

// ---------------------------------------------------------------------
// Micro difficulty: base-layer Local Intrinsic Dimensionality (LID)
// Levina-Bickel MLE estimator.
// ---------------------------------------------------------------------

// d_1 <= ... <= d_K must be sorted ascending, all strictly positive
// (or 0 for the closest point, which is excluded from the sum below).
inline double micro_difficulty(const std::vector<double>& sorted_distances) {
    const size_t K = sorted_distances.size();
    if (K < 2) {
        throw std::invalid_argument("need at least 2 candidate distances for LID");
    }
    double d_K = sorted_distances[K - 1];
    if (d_K <= 0.0) {
        // degenerate: all candidates at distance 0 (exact duplicates)
        return 0.0;
    }

    double sum_log = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < K - 1; ++i) {
        double d_i = sorted_distances[i];
        if (d_i <= 0.0) {
            continue; // skip zero-distance (duplicate) points to avoid log(inf)
        }
        sum_log += std::log(d_K / d_i);
        ++count;
    }
    if (count == 0) {
        return 0.0;
    }
    double mean_log = sum_log / static_cast<double>(count);
    if (mean_log <= 0.0) {
        return 0.0;
    }
    return 1.0 / mean_log;
}

// Convenience wrapper computing both scores from a single trace.
struct DifficultyScores {
    double M; // macro
    double m; // micro
};

inline DifficultyScores compute_difficulty(const SearchTrace& trace,
                                             double eps = 1e-9) {
    DifficultyScores out;
    out.M = macro_difficulty(trace, eps);
    out.m = micro_difficulty(trace.base_layer_distances);
    return out;
}

} // namespace lidef
