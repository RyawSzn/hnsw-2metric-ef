// ada-ef

// remove some reducdant codes and variables like covariance matrices, distance matrices, etc. that are not used in the estimation process, or like some useless timings while keeping the current logic unchanged
#pragma once

#include <Eigen/Core>
#include <memory>
#include <thread>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace hnswdis
{
    // Typedefs for row-major matrices
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using RowVectorXf = Eigen::RowVectorXf;
    using RowVectorXi = Eigen::RowVectorXi;

    class ApproximatedScoreCalculator
    {
    private:
        float truncation_ratio;

    public:
        ApproximatedScoreCalculator(
            float truncation_ratio = 0.25f
        ) : truncation_ratio(truncation_ratio)
        {
        }

        float compute_score(const void *q, const size_t q_size,
                            const void *dist_list, const size_t dist_list_size) const
        {
            return calculate_score(dist_list, dist_list_size);
        }

    private:
        float calculate_score(
            const void *dist_list, const size_t n) const
        {
            if (n == 0) return 0.0f;

            // Cast dist_list to std::pair<float, bool>*
            using edge_t = std::pair<float, bool>;
            const edge_t *edges = static_cast<const edge_t *>(dist_list);

            // Copy and sort the edges ascending by distance
            std::vector<edge_t> E(edges, edges + n);
            std::sort(E.begin(), E.end(), [](const edge_t &a, const edge_t &b) {
                return a.first < b.first;
            });

            // Implement Truncation
            size_t top_n = static_cast<size_t>(n * truncation_ratio);
            if (top_n == 0) top_n = 1;

            // Compute Revisit Rank (R_v)
            float gamma = 16.0f;
            float factor = std::exp(-gamma / static_cast<float>(top_n));
            float w = factor;
            float sum_wv = 0.0f;
            float sum_w = 0.0f;

            for (size_t i = 0; i < top_n; ++i)
            {
                if (E[i].second) // if v_i == 1
                {
                    sum_wv += w;
                }
                sum_w += w;
                w *= factor;
            }

            float r_v = 100.0f * sum_wv / std::max(1e-5f, sum_w);
            return r_v;
        }
    };
}
