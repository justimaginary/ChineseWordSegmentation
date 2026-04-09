import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

dataAll = ["./data/train/as_training.utf8", "./data/train/cityu_training.utf8",
           "./data/train/msr_training.utf8", "./data/train/pku_training.utf8"]


class Vocab:
    def __init__(self):
        # 初始化特殊字符
        # UNK_TAG: 遇到没见过的字就用这个
        # PAD_TAG: 句子太短用来凑数的占位符
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'

        # 建立两个字典：字到数字(char2id)，数字到字(id2char)
        self.char2id = {self.PAD: 0, self.UNK: 1}
        self.id2char = {0: self.PAD, 1: self.UNK}

        # 标签字典（BMES 只有这四种，可以直接写死）
        self.tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '<START>': 4, '<STOP>': 5}
        self.id2tag = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '<START>', 5: '<STOP>'}

    @property
    def vocabSize(self):
        return len(self.char2id)

    def build_vocab(self, filePaths):
        # 遍历清洗好的那四个 utf-8 文件
        # 提取每一行的“字”，如果这个字不在 self.char2id 里，
        # 就给它分配一个新的自增 ID，并存入字典。

        for filename in filePaths:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) != 2:
                        continue
                    if parts[0] not in self.char2id:
                        newId = len(self.char2id)
                        self.char2id[parts[0]] = newId
                        self.id2char[newId] = parts[0]

    def sentence2id(self, sentence):
        # 把汉字列表变成数字列表，遇到不认识的字用 UNK 的 ID
        ans = []
        for char in sentence:
            if char not in self.char2id:
                ans.append(self.char2id[self.UNK])
            else:
                ans.append(self.char2id[char])

        return ans


class CRFDataset(Dataset):
    def __init__(self, dataPaths, vocab):
        self.vocab = vocab
        self.sentences = []  # 存放所有的句子，例如：[['同', '济'], ['大', '学']]
        self.tags = []  # 存放对应的标签，例如：[['B', 'E'], ['B', 'E']]

        currentSentence = []
        currentTags = []

        if isinstance(dataPaths, str):
            dataPaths = [dataPaths]

        for dataPath in dataPaths:
            currentSentence = []
            currentTags = []

            with open(dataPath, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if len(currentSentence) > 0:
                            # 把装满的小推车倒进大仓库
                            self.sentences.append(currentSentence)
                            self.tags.append(currentTags)
                            # 清空小推车，准备装下一句话
                            currentSentence = []
                            currentTags = []
                        continue

                    # 如果不是空行，把字和标签切开，装进临时小推车
                    parts = line.split('\t')
                    if len(parts) == 2:
                        currentSentence.append(parts[0])
                        currentTags.append(parts[1])

                if len(currentSentence) > 0:
                    self.sentences.append(currentSentence)
                    self.tags.append(currentTags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # 从大仓库里拿出第 index 个句子和对应的标签
        sentence = self.sentences[index]
        tags = self.tags[index]

        # sentence 变成数字列表 sentenceIds
        sentenceIds = self.vocab.sentence2id(sentence)

        # tags 变成数字列表 tagIds
        tagIds = [self.vocab.tag2id[tag] for tag in tags]

        # 返回装换好的数字序列
        return sentenceIds, tagIds


def collateFn(batch):
    sentences = []
    tags = []

    for sentence, tag in batch:
        sentences.append(torch.tensor(sentence, dtype=torch.long))
        tags.append(torch.tensor(tag, dtype=torch.long))

    paddedSentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    paddedTags = pad_sequence(tags, batch_first=True, padding_value=0)
    return paddedSentences, paddedTags
