import re

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'r') as f:
    text = f.read()

old_loop = """                while (expected_recall - latest_agg_recall > 1e-4f && latest_ef < max_ef) {
                    float recall_diff = latest_agg_recall - prev_agg_recall;

                    if (recall_diff < 1e-5f && latest_agg_recall >= expected_recall - 1e-3f) {
                        // std::cout << "Recall diff is too small, break." << std::endl;
                        break;
                    }

                    // Protect against division by zero if recall mathematically flatlines before target
                    if (recall_diff < 1e-6f) {
                        break;
                    }

                    // The dynamic proportional guess"""

new_loop = """                while (latest_agg_recall < expected_recall && latest_ef < max_ef) {
                    float recall_diff = latest_agg_recall - prev_agg_recall;

                    // If recall improves by less than epsilon (e.g., 0.001), the search has plateaued.
                    // Break immediately to prevent chasing ghosts and slamming into max_ef.
                    if (recall_diff < epsilon) {
                        break;
                    }

                    // The dynamic proportional guess"""

text = text.replace(old_loop, new_loop)

# Let's also set epsilon to 0.005f to properly fix the 5000 bump as discussed
text = text.replace('float epsilon = 1e-6f;', 'float epsilon = 0.005f;')

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'w') as f:
    f.write(text)

print("Loop optimized!")
