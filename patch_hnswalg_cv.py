import re

with open('hnswlib/hnswalg.h', 'r') as f:
    content = f.read()

# 1. Change return type of adaptiveSearchBaseLayerST
content = content.replace("std::pair<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>, float>\\n    adaptiveSearchBaseLayerST(", "std::tuple<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>, float, float>\\n    adaptiveSearchBaseLayerST(")

# 2. Change return type of adaptiveSearchBaseLayerST2
content = content.replace("std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>\\n    adaptiveSearchBaseLayerST2(", "std::tuple<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>, float, float>\\n    adaptiveSearchBaseLayerST2(")

# 3. Remove d_ep from signatures
content = content.replace("hnswdis::Sketch* sketch,\n        dist_t d_ep = 0,\n        BaseFilterFunctor* isIdAllowed", "hnswdis::Sketch* sketch,\n        BaseFilterFunctor* isIdAllowed")

# 4. Replace d_ep in the score/estimation calculation and add CV calculation
old_calc1 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, d_ep); // used for estimating ef"""

new_calc1 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        float cv = 0.0f;
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
                                cv = std::sqrt(variance) / mean;
                            }
                        }
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, cv); // used for estimating ef"""

content = content.replace(old_calc1, new_calc1)

old_calc2 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, d_ep);  // get estimated ef"""

new_calc2 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        float cv = 0.0f;
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
                                cv = std::sqrt(variance) / mean;
                            }
                        }
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, cv);  // get estimated ef"""

content = content.replace(old_calc2, new_calc2)

# 5. Return tuple
content = content.replace("return {top_candidates, score};", "return {top_candidates, score, cv};")
content = content.replace("return top_candidates;", "return {top_candidates, score, cv};")

# Also need to declare float cv = 0.0f; at the start of adaptiveSearchBaseLayerST and ST2 if it wasn't there so it can be returned
content = content.replace("float score = 0.0;", "float score = 0.0;\\n        float cv = 0.0;")

# Oh wait, my new_calc re-declares `float cv = 0.0f;`. I should just use the outer `cv`.
content = content.replace("float cv = 0.0f;\\n                        if (top_candidates.size() > 1)", "if (top_candidates.size() > 1)")

# 6. Fix adaptiveSearchKnn signature and return
content = content.replace("std::pair<std::priority_queue<std::pair<dist_t, labeltype>>, float>\\n    adaptiveSearchKnn(", "std::tuple<std::priority_queue<std::pair<dist_t, labeltype>>, float, float>\\n    adaptiveSearchKnn(")
# It currently returns `return {result, 0};` at the top
content = content.replace("if (cur_element_count == 0) return {result, 0};", "if (cur_element_count == 0) return {result, 0.0f, 0.0f};")

# Change how it calls adaptiveSearchBaseLayerST
content = content.replace("auto ret = adaptiveSearchBaseLayerST<true>(", "auto ret = adaptiveSearchBaseLayerST<true>(") # no change to call text
content = content.replace("currObj, query_data, std::max(ef_, k), statics_length, score_calculator, sketch, curdist, isIdAllowed", "currObj, query_data, std::max(ef_, k), statics_length, score_calculator, sketch, isIdAllowed")

# Change return of adaptiveSearchKnn
content = content.replace("return {result, ret.second};", "return {result, std::get<1>(ret), std::get<2>(ret)};")

# 7. Fix adaptiveSearchKnnTest
# It doesn't return score/cv, just the queue.
content = content.replace("top_candidates = adaptiveSearchBaseLayerST2<true>(", "top_candidates = std::get<0>(adaptiveSearchBaseLayerST2<true>(")
content = content.replace("top_candidates = adaptiveSearchBaseLayerST2<false>(", "top_candidates = std::get<0>(adaptiveSearchBaseLayerST2<false>(")

with open('hnswlib/hnswalg.h', 'w') as f:
    f.write(content)
