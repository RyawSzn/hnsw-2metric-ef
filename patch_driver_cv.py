with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

content = content.replace("get_dep_centers()", "get_cv_centers()")
content = content.replace("get_dep_tables()", "get_cv_tables()")

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(content)

with open('experiments_driver/util.h', 'r') as f:
    content = f.read()

content = content.replace("get_dep_centers()", "get_cv_centers()")
content = content.replace("get_dep_tables()", "get_cv_tables()")

# update hnsw_search_and_score to return tuple instead of pair if needed?
# hnsw_search_and_score uses ret.first and ret.second. 
# ret is now a tuple! So we need to change it to std::get<0>(ret) and std::get<1>(ret)
content = content.replace("auto &pq = ret.first;", "auto &pq = std::get<0>(ret);")
content = content.replace("result.push_back({labels, ret.second});", "result.push_back({labels, std::get<1>(ret)});")

with open('experiments_driver/util.h', 'w') as f:
    f.write(content)
