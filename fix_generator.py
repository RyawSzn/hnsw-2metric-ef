import re

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'r') as f:
    text = f.read()

old_loop = """                float epilson = 1e-5f;

                while (expected_recall - latest_agg_recall > epilson && latest_ef < max_ef) {
                    // The dynamic proportional guess
                    int step = (int)(ef_diff * (expected_recall - latest_agg_recall) / (latest_agg_recall - prev_agg_recall));
                    int min_step = (int)(k_val * 0.5);
                    ef_diff = std::max(step, min_step);"""

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

# The file might have the exact `old_loop` or slight variations.
# Let's just use regex to target the while loop body safely.
text = re.sub(r'float epilson = 1e-5f;\s*while \(expected_recall - latest_agg_recall > epilson && latest_ef < max_ef\) \{\s*// The dynamic proportional guess\s*int step = \(int\)\(ef_diff \* \(expected_recall - latest_agg_recall\) / \(latest_agg_recall - prev_agg_recall\)\);\s*int min_step = \(int\)\(k_val \* 0\.5\);\s*ef_diff = std::max\(step, min_step\);', new_loop, text)

# Just in case the `old_logic` from my previous thought is actually what's there
text = re.sub(r'while \(expected_recall - latest_agg_recall > 1e-4f && latest_ef < max_ef\) \{\s*float recall_diff = latest_agg_recall - prev_agg_recall;\s*if \(recall_diff < 1e-5f && latest_agg_recall >= expected_recall - 1e-3f\) \{\s*// std::cout << "Recall diff is too small, break." << std::endl;\s*break;\s*\}\s*// Protect against division by zero if recall mathematically flatlines before target\s*if \(recall_diff < 1e-6f\) \{\s*break;\s*\}\s*// The dynamic proportional guess\s*int step = \(int\)\(ef_diff \* \(expected_recall - latest_agg_recall\) / recall_diff\);\s*int min_step = \(int\)\(k_val \* 0\.5\);\s*ef_diff = std::max\(step, min_step\);', new_loop, text)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/table_generator.h', 'w') as f:
    f.write(text)

print("Generator fixed")
