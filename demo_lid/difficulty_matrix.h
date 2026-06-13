#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace lidef {

constexpr int GRID = 100;        // fine grid resolution (B_M, B_m in [0,99])
constexpr int COARSE_GRID = 10;  // coarse fallback grid (10x10 blocks)
constexpr int COARSE_BLOCK = GRID / COARSE_GRID; // = 10

// A single (ef, recall) sample point on a cell's curve.
struct EfRecallPoint {
    int ef;
    double recall;
};

// One cell: a recall curve over a fixed ef schedule, plus sample count
// for sparsity-aware fallback.
struct Cell {
    std::vector<EfRecallPoint> curve; // sorted by ef ascending
    size_t sample_count = 0;

    bool empty() const { return curve.empty(); }
};

// ---------------------------------------------------------------------
// Training-time accumulator: feed (B_M, B_m, recall-vector) per query,
// then call finalize() to produce the smoothed, monotone matrix.
// ---------------------------------------------------------------------
class DifficultyMatrixBuilder {
public:
    // ef_schedule: the fixed budgets swept during training, e.g.
    // {50,100,150,200,300,500,800}, ascending.
    explicit DifficultyMatrixBuilder(std::vector<int> ef_schedule,
                                      size_t sparse_threshold = 20)
        : ef_schedule_(std::move(ef_schedule)), tau_(sparse_threshold) {
        if (ef_schedule_.empty()) {
            throw std::invalid_argument("ef_schedule must be non-empty");
        }
        for (size_t i = 1; i < ef_schedule_.size(); ++i) {
            if (ef_schedule_[i] <= ef_schedule_[i - 1]) {
                throw std::invalid_argument("ef_schedule must be strictly increasing");
            }
        }
        fine_sum_.assign(GRID, std::vector<std::vector<double>>(
                                    GRID, std::vector<double>(ef_schedule_.size(), 0.0)));
        fine_count_.assign(GRID, std::vector<size_t>(GRID, 0));

        coarse_sum_.assign(COARSE_GRID, std::vector<std::vector<double>>(
                                             COARSE_GRID, std::vector<double>(ef_schedule_.size(), 0.0)));
        coarse_count_.assign(COARSE_GRID, std::vector<size_t>(COARSE_GRID, 0));

        global_sum_.assign(ef_schedule_.size(), 0.0);
        global_count_ = 0;
    }

    // recalls.size() must equal ef_schedule_.size(), recalls[k] = recall
    // achieved at ef_schedule_[k] for this query.
    void add_sample(int B_M, int B_m, const std::vector<double>& recalls) {
        if (B_M < 0 || B_M >= GRID || B_m < 0 || B_m >= GRID) {
            throw std::invalid_argument("B_M, B_m must be in [0, 99]");
        }
        if (recalls.size() != ef_schedule_.size()) {
            throw std::invalid_argument("recalls size must match ef_schedule size");
        }

        for (size_t k = 0; k < recalls.size(); ++k) {
            fine_sum_[B_M][B_m][k] += recalls[k];
            global_sum_[k] += recalls[k];
        }
        fine_count_[B_M][B_m] += 1;
        global_count_ += 1;

        int cM = B_M / COARSE_BLOCK;
        int cm = B_m / COARSE_BLOCK;
        for (size_t k = 0; k < recalls.size(); ++k) {
            coarse_sum_[cM][cm][k] += recalls[k];
        }
        coarse_count_[cM][cm] += 1;
    }

    // Produces the final 100x100 matrix: fallback-filled, then both
    // monotonicity passes applied.
    std::vector<std::vector<Cell>> finalize() const;

    const std::vector<int>& ef_schedule() const { return ef_schedule_; }

private:
    std::vector<int> ef_schedule_;
    size_t tau_;

    // fine_sum_[i][j][k]   = sum of recall_k over queries in cell (i,j)
    // fine_count_[i][j]    = number of queries in cell (i,j)
    std::vector<std::vector<std::vector<double>>> fine_sum_;
    std::vector<std::vector<size_t>> fine_count_;

    std::vector<std::vector<std::vector<double>>> coarse_sum_;
    std::vector<std::vector<size_t>> coarse_count_;

    std::vector<double> global_sum_;
    size_t global_count_ = 0;

    Cell make_cell_from_mean(const std::vector<double>& sum, size_t count) const {
        Cell c;
        c.sample_count = count;
        if (count == 0) return c; // empty, caller must fallback further
        c.curve.reserve(ef_schedule_.size());
        for (size_t k = 0; k < ef_schedule_.size(); ++k) {
            c.curve.push_back({ef_schedule_[k], sum[k] / static_cast<double>(count)});
        }
        return c;
    }
};

inline std::vector<std::vector<Cell>> DifficultyMatrixBuilder::finalize() const {
    std::vector<std::vector<Cell>> T(GRID, std::vector<Cell>(GRID));

    // Global fallback cell (used only if even coarse cells are empty).
    Cell global_cell = make_cell_from_mean(global_sum_, global_count_);

    // Stage 1+2: fine cell if sample_count >= tau, else coarse, else global.
    for (int i = 0; i < GRID; ++i) {
        for (int j = 0; j < GRID; ++j) {
            size_t fine_n = fine_count_[i][j];
            if (fine_n >= tau_) {
                T[i][j] = make_cell_from_mean(fine_sum_[i][j], fine_n);
                continue;
            }

            int ci = i / COARSE_BLOCK;
            int cj = j / COARSE_BLOCK;
            size_t coarse_n = coarse_count_[ci][cj];
            if (coarse_n >= tau_) {
                T[i][j] = make_cell_from_mean(coarse_sum_[ci][cj], coarse_n);
                continue;
            }

            // last resort
            T[i][j] = global_cell;
        }
    }

    // --- Monotonicity pass (a): within-cell, recall non-decreasing in ef.
    // Running maximum over each cell's curve.
    for (int i = 0; i < GRID; ++i) {
        for (int j = 0; j < GRID; ++j) {
            auto& curve = T[i][j].curve;
            if (curve.empty()) continue;
            double running_max = curve[0].recall;
            for (size_t k = 1; k < curve.size(); ++k) {
                running_max = std::max(running_max, curve[k].recall);
                curve[k].recall = running_max;
            }
        }
    }

    // --- Monotonicity pass (b): across-cell, recall non-increasing with
    // difficulty at fixed ef. Process in order of increasing i+j, taking
    // min with the cell above (i-1,j) and to the left (i,j-1).
    const size_t K = ef_schedule_.size();
    for (int s = 0; s <= 2 * (GRID - 1); ++s) {
        for (int i = std::max(0, s - (GRID - 1)); i <= std::min(GRID - 1, s); ++i) {
            int j = s - i;
            if (j < 0 || j >= GRID) continue;
            if (i == 0 && j == 0) continue;

            auto& curve = T[i][j].curve;
            if (curve.empty()) continue; // shouldn't happen after fallback, but guard

            for (size_t k = 0; k < K; ++k) {
                double val = curve[k].recall;
                if (i > 0 && !T[i - 1][j].curve.empty()) {
                    val = std::min(val, T[i - 1][j].curve[k].recall);
                }
                if (j > 0 && !T[i][j - 1].curve.empty()) {
                    val = std::min(val, T[i][j - 1].curve[k].recall);
                }
                curve[k].recall = val;
            }
        }
    }

    return T;
}

} // namespace lidef
