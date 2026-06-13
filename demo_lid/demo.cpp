// Demo / smoke test for the LID-EF pipeline.
//
// This file simulates "training" with synthetic search traces (since we
// don't have a real HNSW index here), builds the difficulty matrix, and
// runs an inference-time prediction. Replace the synthetic trace generator
// with calls into your actual HNSW probe search.
#include <iostream>
#include <random>
#include <vector>
#include "difficulty.h"
#include "percentile.h"
#include "difficulty_matrix.h"
#include "predictor.h"

using namespace lidef;

// --- Synthetic trace generator -------------------------------------------
// Produces a SearchTrace whose "difficulty" is controlled by a hidden
// scalar `hardness` in [0,1]: harder queries visit more nodes per layer
// and make less progress, and have flatter base-layer distance profiles
// (higher LID).
SearchTrace make_synthetic_trace(std::mt19937& rng, double hardness, int L = 5, int K = 20) {
    SearchTrace t;
    t.n_l.resize(L + 1);
    t.b_l.resize(L + 1);

    std::uniform_real_distribution<double> jitter(0.9, 1.1);

    double dist = 10.0; // distance shrinks as we descend layers
    for (int l = L; l >= 0; --l) {
        // harder queries visit more nodes
        int base_nodes = 5 + static_cast<int>(hardness * 40);
        t.n_l[l] = static_cast<size_t>(base_nodes * jitter(rng));

        // harder queries make less progress per layer (smaller shrink)
        double shrink = 0.9 - 0.6 * hardness; // hardness=0 -> shrink 0.9 (big drop), hardness=1 -> 0.3
        t.b_l[l] = dist;
        dist *= shrink * jitter(rng);
    }

    // Base layer candidate distances: harder (higher LID) => distances
    // grow more slowly (flatter), easier => distances grow quickly (sharper).
    t.base_layer_distances.resize(K);
    double d0 = 1.0;
    double growth = 1.02 + 0.25 * (1.0 - hardness); // easier -> faster growth -> lower LID
    for (int i = 0; i < K; ++i) {
        t.base_layer_distances[i] = d0 * std::pow(growth, i) * jitter(rng);
    }
    std::sort(t.base_layer_distances.begin(), t.base_layer_distances.end());

    return t;
}

// Simulated recall(ef) for a query of given hardness: a saturating curve
// that approaches 1.0 more slowly for harder queries.
double simulated_recall(double hardness, int ef) {
    double rate = 0.02 * (1.0 - 0.85 * hardness); // harder -> smaller rate -> slower saturation
    double recall = 1.0 - std::exp(-rate * ef);
    return std::min(recall, 1.0);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> hardness_dist(0.0, 1.0);

    const int N = 5000; // training queries
    const std::vector<int> ef_schedule = {20, 50, 100, 150, 200, 300, 500, 800};

    // --- Pass 1: collect raw M, m values to build percentile normalizers ---
    std::vector<double> all_M, all_m;
    std::vector<double> hardness_per_query(N);
    std::vector<SearchTrace> traces; // re-used in pass 2
    traces.reserve(N);

    all_M.reserve(N);
    all_m.reserve(N);

    for (int i = 0; i < N; ++i) {
        double h = hardness_dist(rng);
        hardness_per_query[i] = h;
        SearchTrace trace = make_synthetic_trace(rng, h);
        DifficultyScores s = compute_difficulty(trace);
        all_M.push_back(s.M);
        all_m.push_back(s.m);
        traces.push_back(std::move(trace));
    }

    PercentileNormalizer macro_norm(all_M);
    PercentileNormalizer micro_norm(all_m);

    // --- Pass 2: build the difficulty matrix ---
    DifficultyMatrixBuilder builder(ef_schedule, /*sparse_threshold=*/10);

    for (int i = 0; i < N; ++i) {
        DifficultyScores s = compute_difficulty(traces[i]);
        int B_M = macro_norm.bucket(s.M);
        int B_m = micro_norm.bucket(s.m);

        std::vector<double> recalls(ef_schedule.size());
        for (size_t k = 0; k < ef_schedule.size(); ++k) {
            recalls[k] = simulated_recall(hardness_per_query[i], ef_schedule[k]);
        }
        builder.add_sample(B_M, B_m, recalls);
    }

    auto table = builder.finalize();

    LidEfPredictor predictor(macro_norm, micro_norm, table, ef_schedule);

    // --- Inference: test on a few new synthetic queries ---
    std::cout << "hardness\tB_M\tB_m\tpredicted_ef\tout_of_range\n";
    for (double h : {0.05, 0.3, 0.5, 0.7, 0.95}) {
        SearchTrace probe = make_synthetic_trace(rng, h);
        EfPrediction pred = predictor.predict(probe, /*R_t=*/0.95);
        std::cout << h << "\t" << pred.B_M << "\t" << pred.B_m << "\t"
                  << pred.ef << "\t" << (pred.out_of_range ? "yes" : "no") << "\n";
    }

    return 0;
}
