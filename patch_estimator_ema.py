import re

with open('hnswlib/estimator.h', 'r') as f:
    content = f.read()

# Replace the calculate_score function
old_func = """        float calculate_score(
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
        }"""

new_func = """        float calculate_score(
            const void *dist_list, const size_t n) const
        {
            if (n == 0) return 0.0f;

            using edge_t = std::pair<float, bool>;
            const edge_t *edges = static_cast<const edge_t *>(dist_list);

            float ema = edges[0].first;
            float alpha = 0.05f; // smoothing factor
            
            for (size_t i = 1; i < n; ++i)
            {
                ema = alpha * edges[i].first + (1.0f - alpha) * ema;
            }

            // We return -ema so that a smaller distance (closer/easier) gives a HIGHER score,
            // maintaining the same positive correlation direction as the old RV.
            return -ema;
        }"""

if old_func in content:
    content = content.replace(old_func, new_func)
    print("Replaced calculate_score with EMA version.")
else:
    print("Could not find the old calculate_score function.")

with open('hnswlib/estimator.h', 'w') as f:
    f.write(content)
