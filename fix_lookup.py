with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/lookuptable.h', 'r') as f:
    text = f.read()

# We know the last line is `};` before `} // namespace hnsw_2metric`
text = text.replace("""    }
};

} // namespace hnsw_2metric""", """    }

    int get_average_ef() const {
        if (bins.empty()) return default_ef;
        double sum = 0.0;
        int count = 0;
        for (const auto& bin : bins) {
            if (bin.query_curves.empty()) continue;
            double bin_sum = 0.0;
            int bin_count = 0;
            for (const auto& curve : bin.query_curves) {
                int ef = curve.back().first;
                for (const auto& pt : curve) {
                    if (pt.second >= target_recall) {
                        ef = pt.first;
                        break;
                    }
                }
                bin_sum += ef;
                bin_count++;
            }
            if (bin_count > 0) {
                sum += (bin_sum / bin_count);
                count++;
            }
        }
        return count > 0 ? static_cast<int>(std::round(sum / count)) : default_ef;
    }
};

} // namespace hnsw_2metric""")

# wait, I need to make sure I am not breaking anything from the previous swap_rc_to_ep.py
# since I just `git checkout`'d it, I need to re-run swap_rc_to_ep.py on lookuptable.h!
