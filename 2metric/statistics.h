#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

namespace hnsw_2metric {

/**
 * @brief Analyzes the distribution of dynamically assigned `ef` values.
 * 
 * Prints out how many queries were assigned each specific `ef` value by the 2D grid,
 * matching the intent of `adaptive_ef_analysis` from `experiments_driver`.
 */
inline void adaptive_ef_analysis_2metric(const std::string& dataset, const std::vector<int>& efs_used) {
    std::cout << "\n============================================\n";
    std::cout << "Adaptive EF Analysis for dataset: " << dataset << "\n";
    std::cout << "============================================\n";

    std::map<int, int> ef_distribution;
    for (int ef : efs_used) {
        ef_distribution[ef]++;
    }

    std::cout << std::left << std::setw(10) << "EF Value" << " | " << "Query Count\n";
    std::cout << "-----------------------\n";
    for (const auto& pair : ef_distribution) {
        std::cout << std::left << std::setw(10) << pair.first << " | " << pair.second << "\n";
    }
    std::cout << "============================================\n\n";
}

} // namespace hnsw_2metric
