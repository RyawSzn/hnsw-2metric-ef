// ada-ef
#pragma once

#include <vector>
#include <algorithm>
#include <iostream>

namespace hnswdis
{
    using EfRecallTable = std::vector<std::pair<int, std::vector<std::pair<int, float>>>>;

    class Sketch
    {
    private:
        const EfRecallTable *ef_recall_estimators_single{nullptr};

        const std::vector<EfRecallTable> *tables_{nullptr};
        // cv_centers_[i] is the upper boundary of bucket i (cv < threshold[i] → bucket i).
        // size == tables_.size() - 1; last bucket has no upper bound.
        const std::vector<float>         *cv_centers_{nullptr};

        const float expected_recall;

        std::vector<std::vector<int>> all_links;

        static std::vector<int> build_links(const EfRecallTable &table)
        {
            std::vector<int> links(101);
            int index = 0;
            for (int i = 0; i <= 100; ++i)
            {
                while (index < (int)table.size() && table[index].first < i)
                    ++index;

                int a_index = (index > 0) ? index - 1 : -1;
                int b_index = (index < (int)table.size()) ? index : -1;

                if (a_index != -1 && b_index != -1)
                {
                    links[i] = (std::abs(table[a_index].first - i) <= std::abs(table[b_index].first - i))
                                   ? a_index : b_index;
                }
                else if (a_index != -1) { links[i] = a_index; }
                else                    { links[i] = b_index; }
            }
            return links;
        }

        size_t lookup_ef(const EfRecallTable &table,
                         const std::vector<int> &links,
                         float score) const
        {
            int clamped = static_cast<int>(score);
            if (clamped < 0)   clamped = 0;
            if (clamped > 100) clamped = 100;

            int index = links[clamped];
            const auto &ef_recalls = table[index].second;
            for (const auto &er : ef_recalls)
                if (er.second >= expected_recall)
                    return er.first;
            return ef_recalls.back().first;
        }

        size_t smoothed_ef(const EfRecallTable &table,
                           const std::vector<int> &links,
                           float score) const
        {
            size_t first = lookup_ef(table, links, score);
            if (score < 1 || score >= 100)
                return first;
            return (first + lookup_ef(table, links, score - 1) + lookup_ef(table, links, score + 1)) / 3;
        }

    public:
        Sketch(const EfRecallTable &ef_recall_estimators,
               float expected_recall)
            : ef_recall_estimators_single(&ef_recall_estimators),
              expected_recall(expected_recall)
        {
            all_links.push_back(build_links(ef_recall_estimators));
        }

        Sketch(const std::vector<EfRecallTable> &tables,
               const std::vector<float>         &cv_centers,
               float expected_recall)
            : tables_(&tables),
              cv_centers_(&cv_centers),
              expected_recall(expected_recall)
        {
            all_links.reserve(tables.size());
            for (const auto &t : tables)
                all_links.push_back(build_links(t));
        }

                size_t estimate_ef2(float score, float cv = 0) const
        {
            if (tables_ != nullptr && cv_centers_->size() == tables_->size())
            {
                int n_centers = cv_centers_->size();
                if (n_centers == 1) {
                    return smoothed_ef((*tables_)[0], all_links[0], score);
                }

                if (cv <= (*cv_centers_)[0]) {
                    return smoothed_ef((*tables_)[0], all_links[0], score);
                }
                if (cv >= (*cv_centers_)[n_centers - 1]) {
                    return smoothed_ef((*tables_)[n_centers - 1], all_links[n_centers - 1], score);
                }

                int idx = 0;
                while (idx < n_centers - 1 && cv > (*cv_centers_)[idx + 1]) {
                    idx++;
                }

                float c0 = (*cv_centers_)[idx];
                float c1 = (*cv_centers_)[idx + 1];
                float w = (cv - c0) / (c1 - c0);
                
                size_t ef0 = smoothed_ef((*tables_)[idx], all_links[idx], score);
                size_t ef1 = smoothed_ef((*tables_)[idx + 1], all_links[idx + 1], score);

                return static_cast<size_t>(ef0 * (1.0f - w) + ef1 * w + 0.5f);
            }
            return smoothed_ef(*ef_recall_estimators_single, all_links[0], score);
        }



        size_t estimate_ef(float score) const
        {
            if (tables_ != nullptr)
                return lookup_ef((*tables_)[0], all_links[0], score);
            return lookup_ef(*ef_recall_estimators_single, all_links[0], score);
        }

        size_t get_entry(float score) const
        {
            const EfRecallTable &table = tables_ ? (*tables_)[0] : *ef_recall_estimators_single;
            auto entry = std::lower_bound(table.begin(), table.end(), score,
                                          [](const auto &a, float b) { return a.first < b; });
            if (entry == table.begin())       entry = table.begin();
            else if (entry == table.end())    entry = std::prev(table.end());
            for (const auto &er : entry->second)
                if (er.second >= expected_recall)
                    return er.first;
            return entry->second.back().first;
        }

        void print() const
        {
            auto print_table = [](const EfRecallTable &t) {
                for (const auto &entry : t)
                {
                    std::cout << "Score: " << entry.first << std::endl;
                    for (const auto &p : entry.second)
                        std::cout << "  ef: " << p.first << ", recall: " << p.second << std::endl;
                }
            };

            if (tables_ != nullptr)
            {
                for (int i = 0; i < (int)tables_->size(); ++i)
                {
                    float lo = (i == 0) ? 0.0f : (*cv_centers_)[i - 1];
                    float hi = (i < (int)cv_centers_->size())
                                   ? (*cv_centers_)[i]
                                   : std::numeric_limits<float>::infinity();
                    std::cout << "=== cv bucket " << i
                              << " [" << lo << ", " << hi << ") ===" << std::endl;
                    print_table((*tables_)[i]);
                }
            }
            else { print_table(*ef_recall_estimators_single); }
        }
    };
}
