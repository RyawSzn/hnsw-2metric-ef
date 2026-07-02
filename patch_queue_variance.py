import re

with open('hnswlib/hnswalg.h', 'r') as f:
    content = f.read()

old_logic = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch) {"""

new_logic = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        
                        // --- METHOD 4: Priority Queue Variance (Coefficient of Variation) ---
                        if (top_candidates.size() > 1) {
                            auto temp_q = top_candidates;
                            float sum = 0.0f, sum_sq = 0.0f;
                            int q_size = temp_q.size();
                            while (!temp_q.empty()) {
                                float d = temp_q.top().first;
                                sum += d;
                                sum_sq += d * d;
                                temp_q.pop();
                            }
                            float mean = sum / q_size;
                            if (mean > 1e-6f) {
                                float variance = std::max(0.0f, (sum_sq / q_size) - (mean * mean));
                                float cv = std::sqrt(variance) / mean;
                                // Combine RV and CV. High RV = Easy. High CV = Easy.
                                // We multiply them so both factors compound.
                                score = score * (1.0f + 50.0f * cv); 
                            }
                        }
                        // --------------------------------------------------------------------

                        if (sketch) {"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    print("Patched successfully.")
else:
    print("Could not find the target code to patch.")

with open('hnswlib/hnswalg.h', 'w') as f:
    f.write(content)
