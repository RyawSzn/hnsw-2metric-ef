import re

with open("experiments_driver/run.cpp", "r") as f:
    content = f.read()

# Replace return false back to return in void functions
content = re.sub(
    r'(void offline_laion_text2image\(\).*?\{.*?)(return false;)(.*?)(return false;)(.*?\})',
    r'\1return;\3return;\5',
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'(void compute_groundtruth_laion_text2image\(\).*?\{.*?)(return false;)(.*?)(return false;)(.*?\})',
    r'\1return;\3return;\5',
    content,
    flags=re.DOTALL
)

with open("experiments_driver/run.cpp", "w") as f:
    f.write(content)
