changeData = "./icwb2-data/training/pku_training.utf8"
outputData = "./data/train/pku_training.utf8"


def word2bmes(word):
    length = len(word)
    if length == 0:
        return []
    if length == 1:
        return ['S']
    else:
        tempTag = ['M'] * (length - 2)
        return ['B'] + tempTag + ['E']


with (open(changeData, 'r', encoding='utf8') as fIn,
      open(outputData, 'w', encoding='utf8') as fOut):
    for line in fIn:
        words = line.strip().split()
        if not words:
            continue

        for word in words:
            tags = word2bmes(word)
            for char, tag in zip(word, tags):
                fOut.write(f"{char}\t{tag}\n")

        fOut.write('\n')
