#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

namespace hnsw_2metric {

struct LookupBin {
    float EP_lower;
    float EP_upper;
    float RV_lower;
    float RV_upper;
    int query_count;
    std::vector<std::pair<int, float>> curve;
};

class LookupTable2D {
    std::vector<LookupBin> bins;
    int default_ef;
    float target_recall;

    std::vector<float> ep_bounds;
    std::vector<float> rv_bounds;
    std::vector<std::vector<const LookupBin*>> grid;

    void build_index() {
        if (bins.empty()) return;

        std::vector<float> eps, rvs;
        for (const auto& bin : bins) {
            eps.push_back(bin.EP_lower);
            rvs.push_back(bin.RV_lower);
        }
        std::sort(eps.begin(), eps.end());
        eps.erase(std::unique(eps.begin(), eps.end()), eps.end());
        std::sort(rvs.begin(), rvs.end());
        rvs.erase(std::unique(rvs.begin(), rvs.end()), rvs.end());

        ep_bounds = eps;
        rv_bounds = rvs;

        grid.resize(ep_bounds.size(), std::vector<const LookupBin*>(rv_bounds.size(), &bins[0]));

        for (const auto& bin : bins) {
            auto ep_it = std::lower_bound(ep_bounds.begin(), ep_bounds.end(), bin.EP_lower);
            auto rv_it = std::lower_bound(rv_bounds.begin(), rv_bounds.end(), bin.RV_lower);
            if (ep_it != ep_bounds.end() && rv_it != rv_bounds.end()) {
                int ep_idx = std::distance(ep_bounds.begin(), ep_it);
                int rv_idx = std::distance(rv_bounds.begin(), rv_it);
                grid[ep_idx][rv_idx] = &bin;
            }
        }
    }

public:
    LookupTable2D() : default_ef(50), target_recall(0.95f) {}
    LookupTable2D(const std::vector<LookupBin>& bins_, int default_ef_ = 50, float target_recall_ = 0.95f)
        : bins(bins_), default_ef(default_ef_), target_recall(target_recall_) {
        build_index();
    }

    LookupTable2D(const std::string& csv_path, int default_ef_ = 50, float target_recall_ = 0.95f)
        : default_ef(default_ef_), target_recall(target_recall_) {
        std::ifstream in(csv_path);
        if (!in.is_open()) {
            std::cerr << "Warning: Could not open lookup table " << csv_path << ". Using default_ef=" << default_ef << " for all queries.\n";
            return;
        }

        std::string line;
        std::getline(in, line); 

        auto parse_interval = [](const std::string& s, float& lower, float& upper) {
            size_t p1 = s.find('(');
            size_t p2 = s.find(',');
            size_t p3 = s.find(']');
            if (p1 != std::string::npos && p2 != std::string::npos && p3 != std::string::npos) {
                lower = std::stof(s.substr(p1 + 1, p2 - p1 - 1));
                upper = std::stof(s.substr(p2 + 1, p3 - p2 - 1));
            } else {
                lower = 0.0f; upper = 0.0f;
            }
        };

        while (std::getline(in, line)) {
            if (line.empty()) continue;

            size_t p1 = line.find("\",\"");
            if (p1 == std::string::npos) continue;
            size_t p2 = line.find("\",\"", p1 + 3);
            if (p2 == std::string::npos) continue;
            size_t p3 = line.find("\",\"", p2 + 3);
            if (p3 == std::string::npos) continue;

            std::string ep_str = line.substr(0, p1 + 1);
            std::string rv_str = line.substr(p1 + 3, p2 - (p1 + 3) + 1);
            std::string qc_str = line.substr(p2 + 3, p3 - (p2 + 3));
            std::string curve_str = line.substr(p3 + 4);
            if (!curve_str.empty() && curve_str.back() == '"') {
                curve_str.pop_back();
            }

            LookupBin bin;
            parse_interval(ep_str, bin.EP_lower, bin.EP_upper);
            parse_interval(rv_str, bin.RV_lower, bin.RV_upper);
            bin.query_count = std::stoi(qc_str);

            std::stringstream ss(curve_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                size_t colon = token.find(':');
                if (colon != std::string::npos) {
                    int ef = std::stoi(token.substr(0, colon));
                    float rec = std::stof(token.substr(colon + 1));
                    bin.curve.push_back({ef, rec});
                }
            }
            bins.push_back(bin);
        }
        build_index();
    }

    void set_target_recall(float recall) {
        target_recall = recall;
    }

    int get_ef(float d_ep, float rv) const {
        if (grid.empty()) return default_ef;

        auto ep_it = std::upper_bound(ep_bounds.begin(), ep_bounds.end(), d_ep);
        int ep_idx = std::distance(ep_bounds.begin(), ep_it) - 1;
        if (ep_idx < 0) ep_idx = 0;
        if (ep_idx >= (int)grid.size()) ep_idx = grid.size() - 1;

        auto rv_it = std::upper_bound(rv_bounds.begin(), rv_bounds.end(), rv);
        int rv_idx = std::distance(rv_bounds.begin(), rv_it) - 1;
        if (rv_idx < 0) rv_idx = 0;
        if (rv_idx >= (int)grid[0].size()) rv_idx = grid[0].size() - 1;

        const LookupBin* best_bin = grid[ep_idx][rv_idx];

        if (best_bin->curve.empty()) return default_ef;

        for (const auto& pt : best_bin->curve) {
            if (pt.second >= target_recall) {
                return pt.first;
            }
        }

        return best_bin->curve.back().first;
    }

    int get_average_ef() const {
        if (bins.empty()) return default_ef;
        double total_ef_sum = 0.0;
        int total_queries = 0;
        for (const auto& bin : bins) {
            if (bin.curve.empty() || bin.query_count == 0) continue;
            int ef = bin.curve.back().first;
            for (const auto& pt : bin.curve) {
                if (pt.second >= target_recall) {
                    ef = pt.first;
                    break;
                }
            }
            total_ef_sum += (ef * bin.query_count);
            total_queries += bin.query_count;
        }
        return total_queries > 0 ? static_cast<int>(std::round(total_ef_sum / total_queries)) : default_ef;
    }
};

} // namespace hnsw_2metric
