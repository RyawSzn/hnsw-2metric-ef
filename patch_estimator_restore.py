import re

with open('hnswlib/estimator.h', 'r') as f:
    content = f.read()

old_func = """        float calculate_score(
            const void *dist_list, const size_t n) const
        {
            if (n == 0) return 0.0f;

            using edge_t = std::pair<float, bool>;
            const edge_t *edges = static_cast<const edge_t *>(dist_list);

            float ema_v = 0.0f;
            float alpha = 0.1f; 
            
            for (size_t i = 0; i < n; ++i)
            {
                float v_i = edges[i].second ? 1.0f : 0.0f;
                ema_v = alpha * v_i + (1.0f - alpha) * ema_v;
            }

            // High ema_v means heavily stuck in already-visited nodes (hard query).
            // We return -ema_v so higher score = easier.
            return -ema_v;
        }"""

new_func = """        float calculate_score(
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

content = content.replace(old_func, new_func)

with open('hnswlib/estimator.h', 'w') as f:
    f.write(content)
