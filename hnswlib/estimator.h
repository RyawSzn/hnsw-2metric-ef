// ada-ef
#pragma once

#include <Eigen/Core>
#include <memory>
#include <thread>
#include "boost/math/distributions/normal.hpp"
#include <iostream>
#include <fstream>
#include <omp.h>

namespace hnswdis
{
    // Typedefs for row-major matrices
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using RowVectorXf = Eigen::RowVectorXf;
    using RowVectorXi = Eigen::RowVectorXi;

    // class ScoreCalculator
    // {

    // private:
    //     int num_bins;
    //     std::vector<float> threshold_z;
    //     std::vector<float> bin_weight;

    //     std::vector<float> get_threshold_z(const int num_bins)
    //     {
    //         std::vector<float> bin_threshold(num_bins + 1);
    //         boost::math::normal_distribution<> normal_dist(0.0, 1.0);
    //         bin_threshold[0] = -std::numeric_limits<float>::infinity();
    //         for (int i = 1; i < num_bins; ++i)
    //         {
    //             float p = static_cast<float>(i) / num_bins;
    //             bin_threshold[i] = boost::math::quantile(normal_dist, p);
    //         }
    //         bin_threshold[num_bins] = std::numeric_limits<float>::infinity();
    //         return bin_threshold;
    //     }

    //     std::vector<float> get_bin_weight(const int num_bins)
    //     {
    //         // Parameters for exponential decay
    //         float P0 = 100.0;         // Initial value
    //         float lambda_decay = 1.0; // Decay constant
    //         std::vector<float> weights(num_bins);
    //         for (int i = 0; i < num_bins; ++i)
    //         {
    //             weights[i] = P0 * std::exp(-lambda_decay * i);
    //         }
    //         std::reverse(weights.begin(), weights.end());
    //         return weights;
    //     }

    //     std::vector<float> estimate_threshold(
    //         const float mean, const float var) const
    //     {
    //         float std = std::sqrt(var);

    //         // Calculate bin intervals
    //         std::vector<float> thresholds(threshold_z.size());

    //         for (size_t i = 0; i < threshold_z.size(); ++i)
    //         {
    //             thresholds[i] = mean + threshold_z[i] * std;
    //         }

    //         return thresholds;
    //     }

    // public:
    //     ScoreCalculator(int num_bins)
    //     {
    //         this->num_bins = num_bins;
    //         threshold_z = get_threshold_z(num_bins);
    //         bin_weight = get_bin_weight(num_bins);
    //     }

    //     float calculate_score(
    //         std::vector<float> &dist_list,
    //         const float mean, const float var) const
    //     {
    //         std::vector<float> bin_thresholds = estimate_threshold(mean, var);

    //         std::vector<int> cnt(num_bins, 0);

    //         for (const float dist : dist_list)
    //         {
    //             auto it = std::upper_bound(bin_thresholds.begin(), bin_thresholds.end(), dist);
    //             int bin_index = std::distance(bin_thresholds.begin(), it) - 1;
    //             cnt[bin_index]++;
    //         }

    //         float score = 0.0;
    //         int total = dist_list.size();
    //         for (int i = 0; i < num_bins; ++i)
    //         {
    //             float weight = static_cast<float>(cnt[i]) * 1.0 / total;
    //             score += weight * bin_weight[i];
    //         }

    //         return score;
    //     }
    // };

    class ApproximatedScoreCalculator
    {
    private:
        int num_bins;
        float quantile_step;
        bool isTopBased = false;
        std::vector<float> threshold_z;
        std::vector<float> bin_weight;

    public:
        enum class WeightDecayType
        {
            None,
            Linear,
            Exponential
        };

    private:
        WeightDecayType decay_type = WeightDecayType::Exponential;

        // Initialize quantile thresholds based on whether higher or lower is better
        void initialize_threshold_z()
        {
            threshold_z.resize(num_bins);
            boost::math::normal_distribution<> normal_dist(0.0, 1.0);

            if (isTopBased)
            {
                for (int i = 0; i < num_bins; ++i)
                {
                    threshold_z[i] = boost::math::quantile(
                        normal_dist,
                        (1 - quantile_step * num_bins) + quantile_step * i); // e.g., 0.995, 0.996, ..., 0.999
                }
            }
            else
            {
                for (int i = 0; i < num_bins; ++i)
                {
                    threshold_z[i] = boost::math::quantile(
                        normal_dist,
                        quantile_step * (i + 1)); // e.g., 0.001, 0.002, ..., 0.005
                }
            }
        }

        // Initialize weights based on decay type; max weight = 100
        void initialize_bin_weight()
        {
            bin_weight.resize(num_bins);
            constexpr float P0 = 100.0;         // Initial value
            constexpr float lambda_decay = 1.0; // Decay constant

            // for (int i = 0; i < num_bins; ++i) // original implementation
            // {
            //     bin_weight[i] = P0 * std::exp(-lambda_decay * i); // descending order, in the case of topbased, it requires to be in ascending order
            // }

            for (int i = 0; i < num_bins; ++i)
            {
                switch (decay_type)
                {
                case WeightDecayType::None:
                    bin_weight[i] = P0;
                    break;

                case WeightDecayType::Linear:
                {
                    // Linearly decaying from 100 → 0
                    float t = static_cast<float>(i) / (num_bins - 1);
                    bin_weight[i] = P0 * (1.0f - t);
                    break;
                }

                case WeightDecayType::Exponential:
                    // Exponential decay, normalized so the first bin = 100
                    bin_weight[i] = P0 * std::exp(-lambda_decay * i);
                    break;
                }
            }

            if (isTopBased)
            {
                std::reverse(std::begin(bin_weight), std::end(bin_weight));
            }
        }

        float calculate_score(
            const void *dist_list, const size_t n,
            const float mean, const float var) const
        {
            // Cast dist_list to std::pair<float, bool>*
            using edge_t = std::pair<float, bool>;
            const edge_t *edges = static_cast<const edge_t *>(dist_list);

            // Copy and sort the edges ascending by distance
            std::vector<edge_t> E(edges, edges + n);
            std::sort(E.begin(), E.end(), [](const edge_t &a, const edge_t &b) {
                return a.first < b.first;
            });

            // Compute Revisit Rank (R_v)
            float gamma = 16.0f;
            float sum_wv = 0.0f;
            float sum_w = 0.0f;

            for (size_t i = 0; i < n; ++i)
            {
                float w = std::exp(-gamma * static_cast<float>(i + 1) / static_cast<float>(n));
                if (E[i].second) // if v_i == 1
                {
                    sum_wv += w;
                }
                sum_w += w;
            }

            float r_v = 100 * sum_wv / std::max(1e-6f, sum_w);
            return r_v;
        }

    public:
        ApproximatedScoreCalculator(
            float quantile_step,
            WeightDecayType decay_type = WeightDecayType::Exponential,
            int num_bins = 5
            // number of bins is 5 by default because we use exponential decay function as default which decays quickly and the weights for more than 5 bins are very small;
        ) : quantile_step(quantile_step), decay_type(decay_type), num_bins(num_bins)
        {
            std::cout << "Start creating ApproximatedScoreCalculator" << std::endl;

            // Old distribution init removed

            std::cout << "ApproximatedScoreCalculator created" << std::endl;
        }

        float compute_score(const void *q, const size_t q_size,
                            const void *dist_list, const size_t dist_list_size) const
        {
            return calculate_score(dist_list, dist_list_size, 0.0f, 0.0f);
        }
    };

    std::string weight_decay_type_to_string(ApproximatedScoreCalculator::WeightDecayType type)
    {
        switch (type)
        {
        case ApproximatedScoreCalculator::WeightDecayType::None:
            return "None";
        case ApproximatedScoreCalculator::WeightDecayType::Linear:
            return "Linear";
        case ApproximatedScoreCalculator::WeightDecayType::Exponential:
            return "Exponential";
        default:
            return "Unknown";
        }
    }
}
