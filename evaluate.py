import os
import torch
from dataset import Vocab
from model import BiLstmCrf
from predict import load_model_and_vocab, predict


def generateAnswerSheet(model, vocab, testFilePath, outputFilePath):
    """
    让模型预测，并把答案写到结果文件里
    """
    print(f"正在处理: {testFilePath}")

    # 打开要写入的文件
    with open(testFilePath, 'r', encoding='utf8') as inFile, \
            open(outputFilePath, 'w', encoding='utf8') as outFile:

        # 逐行读取test
        for line in inFile:
            line = line.strip()
            # 如果是空行，直接在结果上也空一行
            if not line:
                outFile.write("\n")
                continue

            # 调用写好的 predict 函数
            # words 列表里装的就是模型切好的词，比如 ["同济", "大学"]
            words, _ = predict(model, vocab, line)

            # 把切好的词用空格拼起来，写进结果
            segmentedLine = "  ".join(words)
            outFile.write(segmentedLine + "\n")

    print(f"预测完成，结果已保存至: {outputFilePath}")


def main():
    # 1. 使用在 train.py 里训练好的模型
    try:
        model, vocab = load_model_and_vocab()
    except Exception as e:
        print("模型加载失败，请检查是否存在模型！错误信息：", e)
        return

    # 2. 确保有存放结果的文件夹
    outputDir = "./results"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # 3.这里以 PKU 的测试集为例，也可以换成 msr 或 cityu
    testFileName = "pku_test.utf8"

    testFilePath = os.path.join("./testing", testFileName)
    outputFilePath = os.path.join(outputDir, testFileName + ".result")

    # 4. 开始test
    if os.path.exists(testFilePath):
        generateAnswerSheet(model, vocab, testFilePath, outputFilePath)
    else:
        print(f"找不到test文件：{testFilePath}")


if __name__ == "__main__":
    main()
