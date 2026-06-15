with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/CMakeLists.txt', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "add_executable(compare_predictors " in line:
        skip = True
    if skip and line.strip() == ")":
        skip = False
        continue
    if not skip:
        new_lines.append(line)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/CMakeLists.txt', 'w') as f:
    f.writelines(new_lines)
