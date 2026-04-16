import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Vocab, CRFDataset, collateFn
from model import BiLstmCrf


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前训练设备: {device}")
    writer = SummaryWriter('runs/bilstm_crf_experiment')
    # 1. 准备数据
    print("正在建立词典...")
    vocab = Vocab()
    trainFiles = ["./data/train/as_training.utf8", "./data/train/cityu_training.utf8",
                  "./data/train/msr_training.utf8", "./data/train/pku_training.utf8"]
    vocab.build_vocab(trainFiles)
    print(f"词表大小: {vocab.vocabSize}")

    print("正在加载数据集...")
    trainDataset = CRFDataset(trainFiles[:3], vocab)

    # DataLoader 是个打包机，每次从 dataset 里抓 batchSize 句话，并用 collateFn 对齐
    trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True, collate_fn=collateFn, num_workers=4,
                             pin_memory=True)

    # 验证集保持不变，依然使用第 4 个文件（pku）的前 500 句作为模拟考
    valDataset = CRFDataset(trainFiles[3], vocab)
    valDataset.sentences = valDataset.sentences[:3500]
    valDataset.tags = valDataset.tags[:3500]
    valLoader = DataLoader(valDataset, batch_size=128, collate_fn=collateFn, num_workers=4,
                           pin_memory=True)

    #  2. 组装机器
    embeddingDim = 256  # 把每个字翻译成 256 维的向量
    hiddenDim = 512  # LSTM 的脑容量大小

    model = BiLstmCrf(vocab.vocabSize, vocab.tag2id, embeddingDim, hiddenDim)

    model = model.to(device)
    #  3. 准备优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    #  4. 开始训练
    epochs = 100

    patience = 10  # 连续 3 次没进步就停
    bestLoss = float('inf')  # 记录历史上最好的成绩
    noImprovementCount = 0  # 计数器：记录连续几次没进步了
    global_step = 0

    print("开始训练！")
    for epoch in range(epochs):
        model.train()
        totalLoss = 0.0

        # 从打包机里一批一批地拿数据
        for batchIdx, (sentences, tags) in enumerate(trainLoader):

            # 1. 清空优化器的梯度
            single_sentence = sentences.to(device)
            single_tags = tags.to(device)
            mask = (sentences != 0).to(device)

            optimizer.zero_grad()

            # 2. 让模型做题，计算 Loss
            # 为了稳定，我们通常取个平均值：
            loss = model(single_sentence, single_tags, mask)

            # 3. 自动反向求导（找错误原因）
            loss.backward()

            # 4. 优化器更新参数
            optimizer.step()

            # 累加 loss 用于打印查看训练进度
            totalLoss += loss.item()

            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            global_step += 1

            if batchIdx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batchIdx} | Loss: {loss.item():.4f}")

        avg_train_loss = totalLoss / len(trainLoader)
        print(f"=== Epoch {epoch + 1} 结束，平均 Loss: {totalLoss / len(trainLoader):.4f} ===")
        writer.add_scalar('Training/Epoch_Loss', avg_train_loss, epoch)

        # 一个 Epoch 训练完了，开始val
        model.eval()
        valTotalLoss = 0

        # 1. 关掉梯度计算
        with torch.no_grad():
            for valSent, valTag in valLoader:
                valMask = (valSent != 0).to(device)
                # 计算验证集上的 loss 并累加到 valTotalLoss
                valTotalLoss += model.forward(valSent.to(device),
                                              valTag.to(device), valMask).item() * valSent.size(0)

        avgValLoss = valTotalLoss / len(valDataset)
        print(f"验证集平均 Loss: {avgValLoss:.4f}")
        scheduler.step(avgValLoss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr}")

        writer.add_scalar('Validation/Epoch_Loss', avgValLoss, epoch)

        # 2. 判断是否进步
        if avgValLoss < bestLoss:
            # a. 更新最好成绩 bestLoss
            # b. 保存当前最好的模型参数 torch.save(...)
            # c. 将计数器 noImprovementCount 清零
            bestLoss = avgValLoss
            torch.save(model.state_dict(), f"./model/model-{epoch}.pth")
            noImprovementCount = 0
            print("发现更好的模型，已保存！")
        else:
            # a. 计数器 noImprovementCount 加 1
            noImprovementCount += 1
            print(f"没有进步，连续不进步次数: {noImprovementCount}")

        # 3. 执行早停
        if noImprovementCount >= patience:
            print("达到忍耐极限，提前停止训练！")
            break

        model.train()  # val结束，切换回训练模式，准备下一轮

    # 训练结束后，把调好的参数保存下来
    torch.save(model.state_dict(), "bilstm_crf_model.pth")
    print("模型已保存！")


if __name__ == "__main__":
    main()
