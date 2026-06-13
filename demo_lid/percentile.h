#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>

namespace lidef {

// Stores a sorted training-distribution array and provides O(log N)
// percentile-rank lookup, mapping any raw value to a bucket in [0, 99].
class PercentileNormalizer {
public:
    PercentileNormalizer() = default;

    // Build from raw training values (M_1..M_N or m_1..m_N). Sorts internally.
    explicit PercentileNormalizer(std::vector<double> raw_values)
        : sorted_values_(std::move(raw_values)) {
        std::sort(sorted_values_.begin(), sorted_values_.end());
    }

    // B(q) = floor(100 * |{i : value_i <= x}| / N), clamped to [0, 99].
    int bucket(double x) const {
        if (sorted_values_.empty()) return 0;
        // upper_bound gives count of elements <= x is found via
        // upper_bound for "<=", since upper_bound finds first element > x
        size_t count_le = static_cast<size_t>(
            std::upper_bound(sorted_values_.begin(), sorted_values_.end(), x)
            - sorted_values_.begin());
        double frac = static_cast<double>(count_le) /
                      static_cast<double>(sorted_values_.size());
        int b = static_cast<int>(std::floor(100.0 * frac));
        if (b < 0) b = 0;
        if (b > 99) b = 99;
        return b;
    }

    size_t size() const { return sorted_values_.size(); }

private:
    std::vector<double> sorted_values_;
};

} // namespace lidef
