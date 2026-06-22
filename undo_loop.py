import re

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'r') as f:
    text = f.read()

new_loop = """                float epsilon = 0.005f;

                while (latest_agg_recall < expected_recall && latest_ef < max_ef) {
                    float recall_diff = latest_agg_recall - prev_agg_recall;

                    // If recall improves by less than epsilon (e.g., 0.5%), the search has plateaued.
                    // Break immediately to prevent chasing ghosts and slamming into max_ef.
                    if (recall_diff < epsilon) {
                        break;
                    }

                    // The dynamic proportional guess
                    int step = (int)(ef_diff * (expected_recall - latest_agg_recall) / recall_diff);
                    int min_step = (int)(k_val * 0.5);
                    ef_diff = std::max(step, min_step);"""

old_loop = """                while (expected_recall - latest_agg_recall > 1e-4f && latest_ef < max_ef) {
                    float recall_diff = latest_agg_recall - prev_agg_recall;

                    if (recall_diff < 1e-5f && latest_agg_recall >= expected_recall - 1e-3f) {
                        // std::cout << "Recall diff is too small, break." << std::endl;
                        break;
                    }

                    // Protect against division by zero
                    if (recall_diff == 0.0f) {
                        break;
                    }

                    // The dynamic proportional guess
                    int step = (int)(ef_diff * (expected_recall - latest_agg_recall) / recall_diff);
                    int min_step = (int)(k_val * 0.5);
                    ef_diff = std::max(step, min_step);"""

text = text.replace(new_loop, old_loop)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'w') as f:
    f.write(text)

print("Undone!")
