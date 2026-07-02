with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

old_main_1 = """    // dump_glove_score_recall();
    // dump_sift_score_recall();
    offline_exp();          // offline computation of estimator, samplings, and ef-adaptor
    online_exp();           // onine search experiments"""
new_main_1 = """    dump_glove_score_recall();
    // dump_sift_score_recall();
    // offline_exp();          // offline computation of estimator, samplings, and ef-adaptor
    // online_exp();           // onine search experiments"""

content = content.replace(old_main_1, new_main_1)

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(content)
