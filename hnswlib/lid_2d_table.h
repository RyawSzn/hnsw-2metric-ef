#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

namespace hnswlid {

static constexpr int LID2D_FILE_MAGIC = 0x4C494433; // "LID3"

class Lid2DTable {
public:
    static constexpr float DECAY_FACTOR = 10.0f;
    static constexpr int   MAX_DIM      = 100;

    static int lid_to_bucket(float lid) {
        // Maps [0, infinity) -> [0, MAX_DIM) using exponential squashing
        int b = static_cast<int>(MAX_DIM * (1.0f - std::exp(-lid / DECAY_FACTOR)));
        if (b < 0) b = 0;
        if (b >= MAX_DIM) b = MAX_DIM - 1;
        return b;
    }

    Lid2DTable() = default;

    struct Obs { float macro; float micro; int ef; };

    void build_from_observations(const std::vector<Obs>& obs, int ef_min, int ef_max)
    {
        ef_min_ = ef_min;
        ef_max_ = ef_max;

        dim_macro_ = MAX_DIM;
        dim_micro_ = MAX_DIM;

        std::vector<std::vector<int>> cell_efs(dim_macro_ * dim_micro_);

        for (const auto& o : obs) {
            int ci = lid_to_bucket(o.macro);
            int cj = lid_to_bucket(o.micro);
            cell_efs[ci * dim_micro_ + cj].push_back(o.ef);
        }

        grid_.assign(dim_macro_ * dim_micro_, -1);
        for (int idx = 0; idx < dim_macro_ * dim_micro_; ++idx) {
            auto& v = cell_efs[idx];
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());
            std::size_t p = static_cast<std::size_t>(0.95 * v.size());
            if (p >= v.size()) p = v.size() - 1;
            grid_[idx] = v[p];
        }

        fill_sparse_cells(ef_min);
        enforce_monotone(ef_min, ef_max);
    }

    int lookup(float macro_lid, float micro_lid) const noexcept {
        int ci = lid_to_bucket(macro_lid);
        int cj = lid_to_bucket(micro_lid);
        
        if (grid_.empty()) return ef_min_;

        int ef = grid_[ci * dim_micro_ + cj];
        if (ef < ef_min_) ef = ef_min_;
        if (ef > ef_max_) ef = ef_max_;
        return ef;
    }

    void print(std::ostream& os = std::cout) const {
        os << "Lid2DTable " << dim_macro_ << "x" << dim_micro_
           << " ef_min=" << ef_min_ << " ef_max=" << ef_max_ << "\n"
           << "Using exponential squashing (decay=" << DECAY_FACTOR << ")\n";
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Lid2DTable: cannot open " + path);
        int magic = LID2D_FILE_MAGIC;
        out.write(reinterpret_cast<const char*>(&magic),    sizeof(int));
        out.write(reinterpret_cast<const char*>(&ef_min_),  sizeof(int));
        out.write(reinterpret_cast<const char*>(&ef_max_),  sizeof(int));
        out.write(reinterpret_cast<const char*>(&dim_macro_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&dim_micro_), sizeof(int));
        
        out.write(reinterpret_cast<const char*>(grid_.data()),
                  static_cast<std::streamsize>(grid_.size() * sizeof(int)));
        out.close();
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Lid2DTable: cannot open " + path);
        int magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(int));
        if (magic != LID2D_FILE_MAGIC)
            throw std::runtime_error("Lid2DTable: bad magic in " + path);
        in.read(reinterpret_cast<char*>(&ef_min_), sizeof(int));
        in.read(reinterpret_cast<char*>(&ef_max_), sizeof(int));
        in.read(reinterpret_cast<char*>(&dim_macro_), sizeof(int));
        in.read(reinterpret_cast<char*>(&dim_micro_), sizeof(int));
        
        grid_.resize(dim_macro_ * dim_micro_);
        in.read(reinterpret_cast<char*>(grid_.data()),
                static_cast<std::streamsize>(grid_.size() * sizeof(int)));
        in.close();
    }

    bool empty() const noexcept { return grid_.empty(); }
    int  ef_min() const noexcept { return ef_min_; }
    int  ef_max() const noexcept { return ef_max_; }

private:
    void fill_sparse_cells(int default_ef) {
        for (int j = 0; j < dim_micro_; ++j) {
            int last = default_ef;
            for (int i = 0; i < dim_macro_; ++i) {
                if (grid_[i * dim_micro_ + j] == -1)
                    grid_[i * dim_micro_ + j] = last;
                else
                    last = grid_[i * dim_micro_ + j];
            }
        }
        for (int j = 0; j < dim_micro_; ++j) {
            int last = default_ef;
            for (int i = dim_macro_ - 1; i >= 0; --i) {
                if (grid_[i * dim_micro_ + j] == -1)
                    grid_[i * dim_micro_ + j] = last;
                else
                    last = grid_[i * dim_micro_ + j];
            }
        }
        for (int i = 0; i < dim_macro_; ++i) {
            int last = default_ef;
            for (int j = 0; j < dim_micro_; ++j) {
                if (grid_[i * dim_micro_ + j] == -1)
                    grid_[i * dim_micro_ + j] = last;
                else
                    last = grid_[i * dim_micro_ + j];
            }
        }
    }

    void enforce_monotone(int ef_min, int ef_max) {
        auto clamp = [&](int v) {
            if (v < ef_min) v = ef_min;
            if (v > ef_max) v = ef_max;
            return v;
        };

        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < dim_micro_; ++j) {
                int running = ef_min;
                for (int i = 0; i < dim_macro_; ++i) {
                    grid_[i * dim_micro_ + j] = clamp(std::max(grid_[i * dim_micro_ + j], running));
                    running = grid_[i * dim_micro_ + j];
                }
            }
            for (int i = 0; i < dim_macro_; ++i) {
                int running = ef_min;
                for (int j = 0; j < dim_micro_; ++j) {
                    grid_[i * dim_micro_ + j] = clamp(std::max(grid_[i * dim_micro_ + j], running));
                    running = grid_[i * dim_micro_ + j];
                }
            }
        }
    }

    int dim_macro_{0};
    int dim_micro_{0};
    std::vector<int> grid_;
    int ef_min_{10};
    int ef_max_{5000};
};

} // namespace hnswlid