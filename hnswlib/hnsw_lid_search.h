#pragma once

// hnsw_lid_search.h — Hierarchical-LID instrumented search
//
// Implements Algorithm A from the Hierarchical-LID Adaptive-EF design:
//
//   Phase 1  (upper layers L_max … 1)
//     Greedy descent, same routing as standard HNSW.
//     Every neighbour distance evaluated is appended to D_top.
//     H_top counts total evaluations across all upper layers.
//
//   Phase 2  (layer 0, initial probe)
//     Run a bounded BFS for exactly BASE_HOPS greedy hops, collecting
//     distances into D_base.
//
//   Phase 3  (compute Triad → Score → ef lookup)
//     m_macro = COMPUTE_LID(D_top)
//     m_micro = COMPUTE_LID(D_base)
//     ∇       = (r_entry − r_base) / max(1, H_top)
//     Score   = floor(w1·m_macro + w2·m_micro − w3·ln(|∇|+ε))
//     ef      = LidEfTable.lookup(Score)   [clamped to ≥ k]
//
//   Phase 4  (standard layer-0 BFS with adaptive ef)
//     Resume layer-0 search using the candidates already in the heap,
//     advancing until the heap-size condition with the new ef is met.
//
// This header is standalone — it does NOT modify hnswalg.h.
// It wraps HierarchicalNSW<float> as a free function that re-uses the index's
// data accessors (all public in hnswlib) to traverse the graph while
// collecting the instrumentation data.

#include "hnswalg.h"
#include "lid_estimator.h"
#include "lid_2d_table.h"

#include <vector>
#include <queue>
#include <utility>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>

namespace hnswlid {

// Result of one instrumented LID search call.
struct LidSearchResult {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> top_k;
    LidTriad  triad;
    int       score{0};
    int       ef_used{0};
};

// ─── instrumented_lid_search ──────────────────────────────────────────────────
//
// Parameters
//   hnsw       : loaded HierarchicalNSW<float> index (read-only)
//   query      : pointer to query vector (must match the index dimension)
//   k          : number of neighbours to return
//   table      : calibrated LID → ef lookup table
//   ef_min     : lower bound on ef (usually equal to k)
//
// Returns a LidSearchResult with the top-k results plus diagnostics.

inline LidSearchResult instrumented_lid_search(
    const hnswlib::HierarchicalNSW<float>& hnsw,
    const float*                            query,
    std::size_t                             k,
    const Lid2DTable&                       table,
    int                                     ef_min = 0)
{
    using tableint   = hnswlib::tableint;
    using dist_t     = float;
    using PairDist   = std::pair<dist_t, tableint>;

    // Comparator: min-heap on distance (smallest distance = best candidate).
    struct CmpMin { bool operator()(const PairDist& a, const PairDist& b) const { return a.first > b.first; } };
    // Max-heap: worst of the top-k candidates sits at top for pruning.
    struct CmpMax { bool operator()(const PairDist& a, const PairDist& b) const { return a.first < b.first; } };

    if (ef_min <= 0) ef_min = static_cast<int>(k);

    LidSearchResult result;

    if (hnsw.cur_element_count == 0) return result;

    // ── Phase 1: upper-layer greedy descent ──────────────────────────────────

    std::vector<float> D_top;
    D_top.reserve(64);

    tableint currObj = hnsw.enterpoint_node_;
    dist_t   curdist = hnsw.fstdistfunc_(
        query,
        hnsw.getDataByInternalId(currObj),
        hnsw.dist_func_param_);

    const dist_t r_entry = curdist;
    int H_top = 0;

    for (int level = hnsw.maxlevel_; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            auto* data = reinterpret_cast<unsigned int*>(
                hnsw.get_linklist(currObj, level));
            int sz = static_cast<int>(hnsw.getListCount(
                reinterpret_cast<hnswlib::linklistsizeint*>(data)));

            auto* nbrs = reinterpret_cast<tableint*>(data + 1);
            for (int i = 0; i < sz; ++i) {
                tableint cand = nbrs[i];
                dist_t d = hnsw.fstdistfunc_(
                    query,
                    hnsw.getDataByInternalId(cand),
                    hnsw.dist_func_param_);
                D_top.push_back(d);
                ++H_top;
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    const dist_t r_base = curdist;

    // ── Phase 2: layer-0 initial probe (BASE_HOPS greedy hops) ───────────────

    std::vector<float> D_base;
    D_base.reserve(BASE_HOPS * 16);

    {
        // Visited bitmap using a simple hash-set style: reuse hnswlib's visited list.
        auto* vl = hnsw.visited_list_pool_->getFreeVisitedList();
        auto* visited  = vl->mass;
        auto  tag      = vl->curV;

        // Seed candidate set with the current best node.
        std::priority_queue<PairDist, std::vector<PairDist>, CmpMin> cand_set;
        std::priority_queue<PairDist, std::vector<PairDist>, CmpMax> top_cands;

        visited[currObj] = tag;
        cand_set.emplace(curdist, currObj);
        top_cands.emplace(curdist, currObj);
        D_base.push_back(curdist);

        int hops_done = 0;
        while (!cand_set.empty() && hops_done < BASE_HOPS) {
            auto [cd, cur_id] = cand_set.top();
            cand_set.pop();
            ++hops_done;

            auto* data = reinterpret_cast<int*>(hnsw.get_linklist0(cur_id));
            int sz = static_cast<int>(hnsw.getListCount(
                reinterpret_cast<hnswlib::linklistsizeint*>(data)));
            auto* nbrs = reinterpret_cast<tableint*>(data + 1);

            for (int j = 0; j < sz; ++j) {
                tableint cand = nbrs[j];
                if (visited[cand] == tag) continue;
                visited[cand] = tag;

                dist_t d = hnsw.fstdistfunc_(
                    query,
                    hnsw.getDataByInternalId(cand),
                    hnsw.dist_func_param_);
                D_base.push_back(d);
                cand_set.emplace(d, cand);
                top_cands.emplace(d, cand);
            }
        }

        // ── Phase 3: Triad → Score → ef ───────────────────────────────────────

        result.triad.macro_lid        = compute_lid(D_top);
        result.triad.micro_lid        = compute_lid(D_base);
        result.triad.descent_gradient = (r_entry - r_base)
                                      / static_cast<float>(std::max(1, H_top));

        int adaptive_ef = table.lookup(result.triad.macro_lid, result.triad.micro_lid);
        adaptive_ef = std::max(adaptive_ef, ef_min);
        adaptive_ef = std::max(adaptive_ef, static_cast<int>(k));
        result.ef_used = adaptive_ef;

        // ── Phase 4: full layer-0 BFS with adaptive ef ────────────────────────
        //
        // Continue directly from the probe's cand_set and top_cands — same
        // visited list (vl), no second allocation, no heap copies.

        dist_t lowerBound = top_cands.empty()
                          ? std::numeric_limits<dist_t>::max()
                          : top_cands.top().first;

        while (!cand_set.empty()) {
            auto [cd2, cur_id2] = cand_set.top();
            if (cd2 > lowerBound && static_cast<int>(top_cands.size()) >= adaptive_ef)
                break;
            cand_set.pop();

            auto* data2 = reinterpret_cast<int*>(hnsw.get_linklist0(cur_id2));
            int sz2 = static_cast<int>(hnsw.getListCount(
                reinterpret_cast<hnswlib::linklistsizeint*>(data2)));
            auto* nbrs2 = reinterpret_cast<tableint*>(data2 + 1);

            for (int j = 0; j < sz2; ++j) {
                tableint cand2 = nbrs2[j];
                if (visited[cand2] == tag) continue;
                visited[cand2] = tag;

                dist_t d2 = hnsw.fstdistfunc_(
                    query,
                    hnsw.getDataByInternalId(cand2),
                    hnsw.dist_func_param_);

                if (static_cast<int>(top_cands.size()) < adaptive_ef || lowerBound > d2) {
                    cand_set.emplace(d2, cand2);
                    if (!hnsw.isMarkedDeleted(cand2))
                        top_cands.emplace(d2, cand2);
                    while (static_cast<int>(top_cands.size()) > adaptive_ef)
                        top_cands.pop();
                    if (!top_cands.empty())
                        lowerBound = top_cands.top().first;
                }
            }
        }

        hnsw.visited_list_pool_->releaseVisitedList(vl);

        // Extract top-k from top_cands.
        while (static_cast<int>(top_cands.size()) > static_cast<int>(k))
            top_cands.pop();

        while (!top_cands.empty()) {
            auto [d, id] = top_cands.top();
            top_cands.pop();
            result.top_k.emplace(d, hnsw.getExternalLabel(id));
        }
    }

    return result;
}

// ─── batch search (convenience wrapper) ──────────────────────────────────────

struct LidBatchStats {
    std::vector<std::vector<std::size_t>> labels;
    std::vector<LidTriad>                 triads;
    std::vector<int>                      scores;
    std::vector<int>                      ef_used;
    long long                             search_ms{0};
};

inline LidBatchStats lid_batch_search(
    const hnswlib::HierarchicalNSW<float>& hnsw,
    const hnswdis::MatrixXf&               query_matrix,
    std::size_t                            k,
    const Lid2DTable&                      table,
    int                                    ef_min = 0)
{
    const int nq = static_cast<int>(query_matrix.rows());
    LidBatchStats stats;
    stats.labels.resize(nq);
    stats.triads.resize(nq);
    stats.scores.resize(nq);
    stats.ef_used.resize(nq);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; ++i) {
        auto r = instrumented_lid_search(
            hnsw, query_matrix.row(i).data(), k, table, ef_min);

        // Extract ordered labels (closest first).
        std::size_t cnt = r.top_k.size();
        stats.labels[i].resize(cnt);
        while (!r.top_k.empty()) {
            stats.labels[i][--cnt] = static_cast<std::size_t>(r.top_k.top().second);
            r.top_k.pop();
        }
        stats.triads[i]  = r.triad;
        stats.scores[i]  = r.score;
        stats.ef_used[i] = r.ef_used;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    return stats;
}

} // namespace hnswlid
