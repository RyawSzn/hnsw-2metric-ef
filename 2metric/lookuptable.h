#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace hnsw_2metric {

/**
 * @struct LookupBin
 * @brief Represents a single bin in the 2D Adaptive EF Lookup Table.
 *
 * Defines the thresholds (lower exclusive, upper inclusive) for RC and RV dimensions,
 * mapping to a specific dynamically assigned `ef` parameter.
 */
struct LookupBin {
    float RC_lower;
    float RC_upper;
    float RV_lower;
    float RV_upper;
    int ef;
};

class LookupTable2D {
    std::vector<LookupBin> bins;
    int default_ef;

public:
    LookupTable2D() : default_ef(50) {}
    LookupTable2D(const std::vector<LookupBin>& bins_, int default_ef_ = 50) : bins(bins_), default_ef(default_ef_) {}

    /**
     * @brief Parses the offline-generated CSV lookup table.
     * @param csv_path The path to the lookup table CSV.
     * @param default_ef_ Fallback `ef` value if a matching bin cannot be found.
     */
    LookupTable2D(const std::string& csv_path, int default_ef_ = 50) : default_ef(default_ef_) {
        std::ifstream in(csv_path);
        if (!in.is_open()) {
            std::cerr << "Warning: Could not open lookup table " << csv_path << ". Using default_ef=" << default_ef << " for all queries.\n";
            return;
        }

        std::string line;
        // Skip header line
        std::getline(in, line);

        while (std::getline(in, line)) {
            if (line.empty()) continue;
            
            // Expected CSV format: "(RC_L,RC_U]","(RV_L,RV_U]","ef","actual_recall"
            size_t p1 = line.find("]\",\"");
            if (p1 == std::string::npos) continue;
            size_t p2 = line.find("]\",\"", p1 + 3);
            if (p2 == std::string::npos) continue;
            size_t p3 = line.find("\",\"", p2 + 3);
            if (p3 == std::string::npos) continue;

            std::string rc_str = line.substr(0, p1 + 1);
            std::string rv_str = line.substr(p1 + 4, p2 - (p1 + 4) + 1);
            std::string ef_str = line.substr(p2 + 4, p3 - (p2 + 4));

            // Lambda to parse standard half-open interval notation
            auto parse_interval = [](const std::string& s, float& lower, float& upper) {
                // Expected token format: (lower,upper] or "(lower,upper]"
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

            LookupBin bin;
            parse_interval(rc_str, bin.RC_lower, bin.RC_upper);
            parse_interval(rv_str, bin.RV_lower, bin.RV_upper);
            bin.ef = std::stoi(ef_str);
            bins.push_back(bin);
        }
    }

    /**
     * @brief Maps computed RC and RV metrics to the optimal `ef`.
     * @param rc The calculated Relative Contrast.
     * @param rv The calculated Topological Rank Entrapment.
     * @return The dynamically assigned `ef` parameter.
     */
    int get_ef(float rc, float rv) const {
        if (bins.empty()) return default_ef;

        for (const auto& bin : bins) {
            // Check containment within half-open interval (lower, upper]
            // We additionally handle the exact 0 bottom bound to ensure inclusivity.
            bool rc_match = (rc > bin.RC_lower && rc <= bin.RC_upper) ||
                            (rc <= bin.RC_lower && bin.RC_lower == 0.0f) ||
                            (rc >= bin.RC_upper && bin.RC_upper >= 1e5f);

            bool rv_match = (rv > bin.RV_lower && rv <= bin.RV_upper) ||
                            (rv <= bin.RV_lower && bin.RV_lower == 0.0f) ||
                            (rv >= bin.RV_upper && bin.RV_upper >= 1e5f);

            if (rc_match && rv_match) {
                return bin.ef;
            }
        }

        // Fallback constraint: If a point falls exactly outside binned edges,
        // calculate the Euclidean distance to the nearest bin center and use its ef.
        float best_dist = 1e9f;
        int best_ef = default_ef;
        for (const auto& bin : bins) {
            float rc_c = (bin.RC_lower + bin.RC_upper) / 2.0f;
            float rv_c = (bin.RV_lower + bin.RV_upper) / 2.0f;
            float dr = rc - rc_c;
            float dv = rv - rv_c;
            float dist = dr*dr + dv*dv;
            if (dist < best_dist) {
                best_dist = dist;
                best_ef = bin.ef;
            }
        }
        return best_ef;
    }
};

} // namespace hnsw_2metric
