with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

bad_main = """    dump_glove_score_recall();
    // dump_sift_score_recall();
    // offline_exp();          // offline computation of estimator, samplings, and ef-adaptor
    // online_exp();           // onine search experiments"""

good_main = """    // dump_glove_score_recall();
    // dump_sift_score_recall();
    offline_exp();          // offline computation of estimator, samplings, and ef-adaptor
    online_exp();           // onine search experiments"""

content = content.replace(bad_main, good_main)

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(content)
