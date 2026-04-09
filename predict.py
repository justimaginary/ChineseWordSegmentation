import torch
from dataset import Vocab
from model import BiLstmCrf


def load_model_and_vocab():
    print("正在重建词典...")
    vocab = Vocab()
    # 必须和训练时的词典完全一致
    trainFiles = ["./data/train/as_training.utf8", "./data/train/cityu_training.utf8",
                  "./data/train/msr_training.utf8", "./data/train/pku_training.utf8"]
    vocab.build_vocab(trainFiles)

    print("正在唤醒模型大脑...")
    embeddingDim = 128
    hiddenDim = 256
    model = BiLstmCrf(vocab.vocabSize, vocab.tag2id, embeddingDim, hiddenDim)

    # 填入训练好的模型路径
    # 如果没改名字，默认加载最后的 bilstm_crf_model.pth
    model.load_state_dict(torch.load("bilstm_crf_model.pth"))
    model.eval()  # 切换到预测模式
    return model, vocab


def predict(model, vocab, text):
    # 1. 汉字转数字
    sentence_ids = vocab.sentence2id(text)
    # 套上 [] 伪装成 batch_size=1
    tensor_sentence = torch.tensor([sentence_ids], dtype=torch.long)

    # 2. 扔进机器找最佳路径
    with torch.no_grad():
        feats = model.getLstmFeatures(tensor_sentence)
        score, tag_ids = model.viterbiDecode(feats[0])

    # 3. 把最佳路径的数字变成 BMES 标签
    tags = [vocab.id2tag[tid] for tid in tag_ids]

    # 4. 把 BMES 标签变成人类可读的词语
    segmented_words = []
    current_word = ""

    for char, tag in zip(text, tags):
        if tag == 'B':
            # 遇到 B，说明遇到了新词的开头。如果还有上一个词的残留，强行打包。
            if current_word:
                segmented_words.append(current_word)
                current_word = ""
            current_word += char

        elif tag == 'M':
            # 遇到 M，继续往推车里塞字
            current_word += char

        elif tag == 'E':
            # 遇到 E，词语结束，打包存入仓库
            current_word += char
            segmented_words.append(current_word)
            current_word = ""

        elif tag == 'S':
            # 遇到 S，单字成词。先清理历史遗留，再把当前字存进去
            if current_word:
                segmented_words.append(current_word)
                current_word = ""
            segmented_words.append(char)

        else:
            # 遇到其他意外标签（容错处理），当单字处理
            if current_word:
                segmented_words.append(current_word)
                current_word = ""
            segmented_words.append(char)

    # 收尾：如果遍历完了，还有字，强行打包
    if current_word:
        segmented_words.append(current_word)

    return segmented_words, tags


if __name__ == "__main__":
    # 这个代码块只在直接运行 predict.py 时执行，用来做单句测试
    try:
        model, vocab = load_model_and_vocab()

        test_text = "我们在同济大学信息安全专业学习"
        print(f"\n==========================================")
        print(f"原句: {test_text}")

        words, tags = predict(model, vocab, test_text)

        print(f"预测标签: {tags}")
        print(f"分词结果: {' / '.join(words)}")
        print(f"==========================================")

    except FileNotFoundError:
        print("找不到模型文件！请确保已经运行过 train.py 并且生成了 .pth 文件。")
