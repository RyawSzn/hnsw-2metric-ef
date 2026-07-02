import re

with open('hnswlib/estimator.h', 'r') as f:
    content = f.read()

old_func = """        float calculate_score(
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

new_func = """        float calculate_score(
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

content = content.replace(old_func, new_func)

with open('hnswlib/estimator.h', 'w') as f:
    f.write(content)
