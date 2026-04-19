input_file = "./data/train/pku_training.utf8"
output_dict = "pku_dict.txt"

words_set = set()
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 训练集原文件是以空格分隔词语的
        words = line.strip().split()
        words_set.update(words)

with open(output_dict, 'w', encoding='utf-8') as f:
    for w in sorted(list(words_set)):
        f.write(w + "\n")