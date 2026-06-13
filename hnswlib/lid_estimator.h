#pragma once

// Hierarchical-LID Adaptive-EF
// ─────────────────────────────────────────────────────────────────────────────
// A purely geometric, distribution-free approach to predicting query
// difficulty and dynamically assigning an optimal ef in O(1).
//
// Three difficulty signals (the Triad):
//   Macro-LID  – computed from D_top (distances seen while routing through
//                upper layers).  Diverges when the query sits on a cluster
//                boundary → strong hardness signal.
//   Micro-LID  – computed from D_base (first BASE_HOPS hops on layer 0).
//                Captures local spatial density.
//   Descent Gradient ∇ – (r_entry − r_base) / max(1, H_top).
//                Detects topological voids / outlier queries whose distance
//                barely shrinks while descending through the hierarchy.
//
// LID MLE formula (applied to a distance array D):
//   LID = -|D| / Σ ln( r_i / (r_max + ε) )
//
// Unified query score:
//   Score = floor( w1·m_macro + w2·m_micro − w3·ln(∇ + ε) )
//
// Lookup table: Score → adaptive_ef
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstddef>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>

namespace hnswlid {

// ─── tuneable hyper-parameters ────────────────────────────────────────────────

// Number of initial hops on layer 0 used to build D_base.
// Mirrors the "M = 10 initial hops" in the pseudocode.
static constexpr int BASE_HOPS = 10;


// Blending weights for the three difficulty signals.
// Sensible defaults calibrated so all three contributions are of similar
// magnitude under typical HNSW configurations (M=16, 2–4 upper layers).
//   w1  amplifies Macro-LID (boundary detection – the strongest signal)
//   w2  amplifies Micro-LID (local density)
//   w3  down-weights the gradient term (logarithm already compresses scale)
static constexpr float W1 = 1.0f;  // Macro-LID weight
static constexpr float W2 = 0.5f;  // Micro-LID weight
static constexpr float W3 = 2.0f;  // Descent-gradient log weight

// Small epsilon – guards log(0) and division by near-zero variance.
static constexpr float LID_EPSILON = 1e-10f;

// ─── core LID MLE ─────────────────────────────────────────────────────────────

// Compute the Maximum Likelihood Estimate of the Local Intrinsic Dimensionality
// from an array of distances (raw, NOT necessarily squared – we use whatever the
// distance function returns, which for hnswlib with InnerProductSpace is a
// cosine-distance value in [0,2]).
//
// Formula:  LID = -n / Σ_i  ln( d_i / (d_max + ε) )
//
// If the array is empty or all distances are identical (variance = 0, which
// happens exactly when the query is on a cluster boundary) the denominator
// collapses to zero.  We clamp the result to a large sentinel value so that
// the Score formula correctly signals a very hard query.
//
// Template parameter lets the caller pass a plain C-array or std::vector data
// pointer without copying.
inline float compute_lid(const float* distances, std::size_t n) noexcept {
    if (n == 0) return 0.0f;

    float d_max = *std::max_element(distances, distances + n);
    float d_max_safe = d_max + LID_EPSILON;

    float sum_log = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float ratio = distances[i] / d_max_safe;
        // ratio is in (0, 1] → log is in (-∞, 0].  We want the negated sum.
        sum_log += std::log(ratio + LID_EPSILON);
    }

    // sum_log ≤ 0 always.  Denominator = -sum_log ≥ 0.
    float denom = -sum_log;
    if (denom < LID_EPSILON)
        return static_cast<float>(n) / LID_EPSILON;
        
    return static_cast<float>(n) / denom;
}

// Convenience overload for std::vector.
inline float compute_lid(const std::vector<float>& distances) noexcept {
    return compute_lid(distances.data(), distances.size());
}

// ─── difficulty triad ─────────────────────────────────────────────────────────

// Aggregated per-query difficulty signals produced by the instrumented search.
struct LidTriad {
    float macro_lid{0.0f};       // from upper-layer routing distances
    float micro_lid{0.0f};       // from first BASE_HOPS on layer 0
    float descent_gradient{0.0f};// (r_entry − r_base) / max(1, H_top)
};

// Compute the unified scalar Score from the triad.
// Score is clamped to [0, ∞) and cast to int for table lookup.
inline int compute_score(const LidTriad& triad) noexcept {
    // ∇ can be negative if the search ascends (shouldn't happen in practice,
    // but the formula handles it gracefully through ln(|∇| + ε)).
    float grad = triad.descent_gradient;
    float log_grad = std::log(std::abs(grad) + LID_EPSILON);

    float raw = W1 * triad.macro_lid
              + W2 * triad.micro_lid
              - W3 * log_grad;

    // Floor and clamp to non-negative integers.
    int score = static_cast<int>(std::floor(raw));
    return (score < 0) ? 0 : score;
}

// ─── ef lookup table ──────────────────────────────────────────────────────────
//
// Maps Score → adaptive_ef.
//
// Design: the table is built offline (once per dataset) during a calibration
// pass.  At query time, table lookup is O(1).
//
// The table stores a flat vector of (score_bucket, ef) pairs sorted by
// score_bucket.  At query time we binary-search for the score and return the
// associated ef.  For scores beyond the table range we return ef_upper_bound.

class LidEfTable {
public:
    // Default-construct an empty table (ef_min returned for all scores).
    LidEfTable() = default;

    // Build a simple linear table: ef grows from ef_min to ef_max linearly
    // over score_max buckets.  This is a reasonable prior before calibration.
    //
    //   ef(s) = ef_min + round( (ef_max − ef_min) · s / score_max )
    //
    // Callers replace this with a calibrated table loaded from disk.
    void build_linear(int ef_min, int ef_max, int score_max) {
        entries_.clear();
        entries_.reserve(score_max + 1);
        for (int s = 0; s <= score_max; ++s) {
            float frac = (score_max > 0)
                       ? static_cast<float>(s) / static_cast<float>(score_max)
                       : 0.0f;
            int ef = ef_min + static_cast<int>(std::round(frac * (ef_max - ef_min)));
            ef = std::clamp(ef, ef_min, ef_max);
            entries_.push_back(Entry{s, ef});
        }
        ef_upper_ = ef_max;
        ef_lower_ = ef_min;
    }

    // Lookup: return the ef for a given score.
    // Binary search → O(log N) where N = number of distinct score buckets.
    // For a dense, sorted table this reduces to O(1) direct indexing.
    int lookup(int score) const noexcept {
        if (entries_.empty()) return ef_lower_;
        if (score <= entries_.front().score) return entries_.front().ef;
        if (score >= entries_.back().score)  return ef_upper_;

        // Binary search for the first entry with bucket ≥ score.
        auto it = std::lower_bound(entries_.begin(), entries_.end(), score,
            [](const Entry& e, int s) { return e.score < s; });
        if (it == entries_.end()) return ef_upper_;
        return it->ef;
    }

    // ── Serialization ──────────────────────────────────────────────────────
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("LidEfTable: cannot open " + path);
        int n = static_cast<int>(entries_.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(&ef_upper_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&ef_lower_), sizeof(int));
        for (const auto& e : entries_) {
            out.write(reinterpret_cast<const char*>(&e.score), sizeof(int));
            out.write(reinterpret_cast<const char*>(&e.ef),    sizeof(int));
        }
        out.close();
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("LidEfTable: cannot open " + path);
        int n = 0;
        in.read(reinterpret_cast<char*>(&n),         sizeof(int));
        in.read(reinterpret_cast<char*>(&ef_upper_), sizeof(int));
        in.read(reinterpret_cast<char*>(&ef_lower_), sizeof(int));
        entries_.resize(n);
        for (auto& e : entries_) {
            in.read(reinterpret_cast<char*>(&e.score), sizeof(int));
            in.read(reinterpret_cast<char*>(&e.ef),    sizeof(int));
        }
        in.close();
    }

    // Direct insertion – used during calibration.
    void insert(int score, int ef) {
        entries_.push_back({score, ef});
        if (ef > ef_upper_) ef_upper_ = ef;
        if (ef < ef_lower_ || ef_lower_ == 0) ef_lower_ = ef;
    }

    void sort_and_deduplicate() {
        std::sort(entries_.begin(), entries_.end(),
            [](const Entry& a, const Entry& b) { return a.score < b.score; });
        entries_.erase(std::unique(entries_.begin(), entries_.end(),
            [](const Entry& a, const Entry& b) { return a.score == b.score; }),
            entries_.end());
    }

    void print(std::ostream& os = std::cout) const {
        os << "LidEfTable (" << entries_.size() << " entries, ef_lower="
           << ef_lower_ << ", ef_upper=" << ef_upper_ << "):\n";
        for (const auto& e : entries_)
            os << "  score=" << e.score << " -> ef=" << e.ef << "\n";
    }

    int ef_upper() const noexcept { return ef_upper_; }
    int ef_lower() const noexcept { return ef_lower_; }
    bool empty()   const noexcept { return entries_.empty(); }

private:
    struct Entry { int score; int ef; };
    std::vector<Entry> entries_;
    int ef_upper_{10};
    int ef_lower_{10};
};

// ─── calibration helper ───────────────────────────────────────────────────────
//
// Given a set of (score, required_ef) calibration observations this builds a
// monotone table where ef is guaranteed to be non-decreasing with score.
// "Required ef" means the minimum ef at which recall ≥ target was observed.

class LidTableCalibrator {
public:
    explicit LidTableCalibrator(float target_recall, int ef_max, int k)
        : target_recall_(target_recall), ef_max_(ef_max), k_(k) {}

    void add_observation(int score, int ef_required) {
        obs_.emplace_back(score, ef_required);
    }

    LidEfTable build() const {
        if (obs_.empty()) {
            LidEfTable t;
            t.build_linear(k_, ef_max_, 100);
            return t;
        }

        // Group by score bucket.
        std::map<int, std::vector<int>> by_score;
        for (const auto& [s, ef] : obs_)
            by_score[s].push_back(ef);

        // For each bucket use the 95th-percentile required-ef (conservative).
        std::vector<std::pair<int,int>> raw;
        for (auto& [s, efs] : by_score) {
            std::sort(efs.begin(), efs.end());
            std::size_t idx = static_cast<std::size_t>(0.95 * efs.size());
            if (idx >= efs.size()) idx = efs.size() - 1;
            raw.emplace_back(s, efs[idx]);
        }
        std::sort(raw.begin(), raw.end());

        // Enforce monotone: ef must not decrease as score increases.
        int running_max = k_;
        LidEfTable table;
        for (auto& [s, ef] : raw) {
            if (ef < running_max) ef = running_max;
            if (ef > ef_max_)    ef = ef_max_;
            running_max = ef;
            table.insert(s, ef);
        }
        table.sort_and_deduplicate();
        return table;
    }

private:
    float target_recall_;
    int ef_max_;
    int k_;
    std::vector<std::pair<int,int>> obs_;
};

} // namespace hnswlid
