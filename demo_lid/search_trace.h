#pragma once
#include <vector>
#include <cstddef>

namespace lidef {

// Populated by the HNSW search routine during a (probe or full) search.
// The caller is responsible for filling these fields from its own graph
// traversal -- this struct is the contract between HNSW internals and
// the difficulty model.
struct SearchTrace {
    // Per-layer statistics, indexed by layer l = 0 .. L (layer 0 = base).
    // n_l[l]  = number of unique nodes evaluated at layer l
    // b_l[l]  = best (minimum) distance to the query found by end of layer l
    std::vector<size_t> n_l;
    std::vector<double> b_l;

    // The K smallest candidate distances encountered during the base-layer
    // probe, sorted ascending: d_1 <= d_2 <= ... <= d_K
    std::vector<double> base_layer_distances;
};

} // namespace lidef
