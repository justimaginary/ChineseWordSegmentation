import torch
import torch.nn as nn

startTag = "<START>"
stopTag = "<STOP>"
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        # 建立一个线性层，用来计算当前字和全句的相关度
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs 形状: [batch_size, seq_len, hidden_dim]
        
        # 1. 计算每个字对全句的重要性得分
        energy = self.projection(encoder_outputs) # [batch_size, seq_len, 1]
        
        # 2. 算出权重分布 (Softmax)
        weights = torch.softmax(energy, dim=1) # [batch_size, seq_len, 1]
        
        # 3. 将权重应用到原特征上，得到上下文向量
        # 利用广播机制进行加权求和
        context = encoder_outputs * weights # [batch_size, seq_len, hidden_dim]
        
        return context

class BiLstmCrf(nn.Module):
    def __init__(self, vocabSize, tagToIx, embeddingDim, hiddenDim):
        super(BiLstmCrf, self).__init__()
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.vocabSize = vocabSize
        self.tagToIx = tagToIx
        self.tagsetSize = len(tagToIx)
        self.dropout = nn.Dropout(0.5)

        # 词向量层
        # nn.Embedding 需要知道一共有多少个字 (vocabSize)，以及要把字翻译成多长的向量 (embeddingDim)
        self.wordEmbeds = nn.Embedding(vocabSize, embeddingDim)

        # LSTM 层
        # bidirectional=True 开启双向阅读
        self.lstm = nn.LSTM(embeddingDim, hiddenDim // 2,
                            num_layers=3, bidirectional=True, batch_first=True)
                            
        # 注意力层
        self.attention = SelfAttention(hiddenDim)
        # 将 LSTM 的输出映射到标签空间
        # 线性映射层：把 LSTM 复杂的输出，压缩成标签的数量（比如 6 个标签就有 6 个分数）
        self.hidden2Tag = nn.Linear(hiddenDim, self.tagsetSize)

        # 转移矩阵 (transition matrix)
        # 转移矩阵 transitions[i, j] 代表从标签 j 转移到标签 i 的分数
        # 因为这是一张需要机器自己学习的表，所以我们要用 nn.Parameter 把它包装起来
        # torch.randn 会随机生成一堆数字来填满这张表，作为初始状态
        self.transitions = nn.Parameter(
            torch.randn(self.tagsetSize, self.tagsetSize)
        )

        # 强制约束：任何标签都不可能转移到 START，STOP 也不可能转移到任何标签
        self.transitions.data[tagToIx[startTag], :] = -1000000
        self.transitions.data[:, tagToIx[stopTag]] = -1000000

    def getLstmFeatures(self, sentence):
        # 提取 LSTM 给出的初步特征（状态分）

        embeds = self.wordEmbeds(sentence)
        lstmOut, _ = self.lstm(embeds)
        
        contextual_out = self.attention(lstmOut) # [batch_size, seq_len, hiddenDim]
        
        fused_out = lstmOut + contextual_out
        
        # 3. 映射到标签空间
        fused_out = self.dropout(fused_out)
        lstmFeats = self.hidden2Tag(fused_out)
        return lstmFeats

    '''
    def scoreSentence(self, feats, tags):
        """
        计算一条真实路径的总得分（分子）
        :param feats: 二楼 LSTM 给出的特征矩阵，大小是 [句子长度, 标签数量]
        :param tags: 这句话真实的标签序列，比如 [0, 2] 代表 [B, E]
        """
        # 初始化总分为 0 (必须用 PyTorch 的 tensor 格式，方便后续求导)
        score = torch.zeros(1,device=feats.device)

        # 为了处理第一次从 START 起跳的逻辑
        # 我们把 START 的 ID 强行拼接到真实 tags 的最前面
        # 假如 tags 原本是 [B, E]，拼接后就变成了 [START, B, E]
        startTensor = torch.tensor([self.tagToIx[startTag]], dtype=torch.long,device=feats.device)
        tags = torch.cat([startTensor, tags])

        # 接下来，开始像超级玛丽一样，一步步闯关计算分数
        # enumerate 会同时拿出当前的步数 i，和当前字在各个标签上的意向分 feat
        for i, feat in enumerate(feats):
            # 注意：这里的 i 对应的是这句话里的第 i 个字
            # 但是在 tags 列表里，因为我们前面强行塞了一个 START
            # 所以当前字对应的真实标签是 tags[i+1]，而它上一步的标签是 tags[i]

            # 计算转移分 (Jump Score)
            # 从上一个标签 tags[i] 跳到当前标签 tags[i+1] 的得分
            jumpScore = self.transitions[tags[i + 1], tags[i]]

            # 计算状态分 (Status Score)
            # 从 feat 这个意向分列表里，拿出当前真实标签 tags[i+1] 对应的那个分数
            statusScore = feat[tags[i + 1]]

            # 累加到总分
            score = score + jumpScore + statusScore

        score = score + self.transitions[self.tagToIx[stopTag], tags[-1]]

        return score
    '''

    def scoreSentence(self, feats, tags, mask):
        """
        Batched 计算真实路径得分
        """
        batch_size, seq_len = tags.shape
        score = torch.zeros(batch_size, device=feats.device)

        # 把 START 拼接到每一句话的最前面
        start_tags = torch.full((batch_size, 1), self.tagToIx[startTag], dtype=torch.long, device=feats.device)
        pad_tags = torch.cat([start_tags, tags], dim=1)  # 形状: [batch_size, seq_len + 1]

        for t in range(seq_len):
            current_tag = pad_tags[:, t + 1]
            previous_tag = pad_tags[:, t]
            mask_t = mask[:, t]

            # 提取当前字的正确意向分 (使用 gather)
            emit_score = feats[:, t, :].gather(1, current_tag.unsqueeze(1)).squeeze(1)
            # 提取转移分
            trans_score = self.transitions[current_tag, previous_tag]

            # 核心：只累加有效字的得分，<PAD> 的得分乘以 0 被抹去
            step_score = (emit_score + trans_score) * mask_t
            score += step_score

        # 寻找每句话真正的最后一个标签 (因为后面跟着一堆 PAD)
        # mask 的和就是这句话的真实长度
        seq_lens = mask.sum(dim=1).long()
        last_tags = pad_tags.gather(1, seq_lens.unsqueeze(1)).squeeze(1)

        # 加上真实的跳向 STOP 的分数
        stop_score = self.transitions[self.tagToIx[stopTag], last_tags]
        score += stop_score

        return score

    '''
    def forwardAlg(self, feats):
        forwardVar = torch.full((1, self.tagsetSize), -1000000.0, device=feats.device)
        forwardVar[0][self.tagToIx[startTag]] = 0.

        for feat in feats:
            al = []
            for nextTag in range(self.tagsetSize):
                # 状态分
                emitScore = feat[nextTag].view(1, -1).expand(1, self.tagsetSize)
                # 转移分
                transScore = self.transitions[nextTag].view(1, -1)

                nextTagVar = forwardVar + emitScore + transScore

                totalScore = torch.logsumexp(nextTagVar, dim=1).view(1)
                al.append(totalScore)

            forwardVar = torch.cat(al).view(1, -1)

        # 加上走到 STOP 标签的转移分
        terminalVar = forwardVar + self.transitions[self.tagToIx[stopTag]]
        # 最后一次揉合，得到终极的分母 Z(x)
        alpha = torch.logsumexp(terminalVar, dim=1)

        return alpha
    '''

    def forwardAlg(self, feats, mask):
        """
        Batched 前向算法
        :param feats: 形状 [batch_size, seq_len, tagsetSize]
        :param mask: 形状 [batch_size, seq_len]，由 1 和 0 组成
        """
        batch_size, seq_len, _ = feats.shape

        # 初始化记分牌 [batch_size, tagsetSize]
        forwardVar = torch.full((batch_size, self.tagsetSize), -1000000.0, device=feats.device)
        forwardVar[:, self.tagToIx[startTag]] = 0.

        # 沿着时间步（句子长度）往前走
        for t in range(seq_len):
            feat = feats[:, t, :]  # 当前时间步的意向分 [batch_size, tagsetSize]
            mask_t = mask[:, t].unsqueeze(1)  # 当前时间步的掩码 [batch_size, 1]

            # 广播魔法：计算所有路径组合的分数
            # forwardVar.unsqueeze(2): [batch_size, num_tags, 1]
            # self.transitions: [num_tags, num_tags]
            # feat.unsqueeze(1): [batch_size, 1, num_tags]
            tagVar = forwardVar.unsqueeze(1) + self.transitions + feat.unsqueeze(2)

            # 算出理论上的下一步总分
            next_forwardVar = torch.logsumexp(tagVar, dim=2)

            # 掩码拦截：如果当前字是有效的 (mask=1)，更新记分牌；如果是 <PAD> (mask=0)，保留上一步的记分牌
            forwardVar = torch.where(mask_t == 1, next_forwardVar, forwardVar)

        # 加上走到 STOP 标签的转移分
        terminalVar = forwardVar + self.transitions[self.tagToIx[stopTag]]
        return torch.logsumexp(terminalVar, dim=1)

    '''
    def forward(self, sentence, tags):
        feats = self.getLstmFeatures(sentence)
        forwardScore = self.forwardAlg(feats)
        goldScore = self.scoreSentence(feats, tags)
        return forwardScore - goldScore
    '''

    def forward(self, sentences, tags, mask):
        # 一二层不变，处理整个 Batch
        feats = self.getLstmFeatures(sentences)

        # 把 mask 传给三层的 CRF
        forwardScore = self.forwardAlg(feats, mask)
        goldScore = self.scoreSentence(feats, tags, mask)

        # 返回整个 Batch 的平均 Loss
        return (forwardScore - goldScore).mean()

    def viterbiDecode(self, feats):
        backpointers = []  # 用来记录每一步的所有内容

        # 初始化记分牌 (和 forwardAlg 一模一样)
        initVvars = torch.full((1, self.tagsetSize), -1000000.,device=feats.device)
        initVvars[0][self.tagToIx[startTag]] = 0

        # forwardVar 现在代表：走到上一步时，各个标签的最高得分
        forwardVar = initVvars

        for feat in feats:
            bptrsT = []  # 记录当前这一步，跳到各个标签的最优“前一步”是谁
            viterbivarsT = []  # 记录当前这一步，跳到各个标签的最高得分

            for nextTag in range(self.tagsetSize):
                # 这一次不需要加上状态分 (emitScore)，
                # 先单独比较：(上一步的最高得分 + 转移到 nextTag 的转移分)
                # 看看从哪里跳过来最划算
                nextTagVar = forwardVar + self.transitions[nextTag]
                # 1. 找出 nextTagVar 里面最大的那个分数 (bestScore)
                # 2. 找出那个最大分数对应的索引 (bestTagId)，也就是它是从哪个标签跳过来的！
                bestScore = torch.max(nextTagVar, dim=1)[0]
                bestTagId = torch.max(nextTagVar, dim=1)[1]


                # 记录：把最好的前一步标签存进本子
                bptrsT.append(bestTagId.item())
                # 记录得分：把最高分存下来
                viterbivarsT.append(bestScore.view(1))

            # 加上当前字的状态分 (Emission Score)
            # 因为状态分对所有跳法都是一样的，所以我们选出最优跳法后再加它
            forwardVar = (torch.cat(viterbivarsT) + feat).view(1, -1)
            # 把这一步的脚印彻底归档到大本子里
            backpointers.append(bptrsT)

        # 循环结束，此时已经遍历完了句子中的每一个字

        # 1. 最后一步：加上走到 STOP 的转移分
        terminalVar = forwardVar + self.transitions[self.tagToIx[stopTag]]

        # 2. 找出到达 STOP 前的最后一步，哪个标签得分最高？
        # 使用 .item() 是为了把 PyTorch 的张量变成 Python 里的普通数字
        bestFinalTagId = torch.max(terminalVar, dim=1)[1].item()
        pathScore = torch.max(terminalVar, dim=1)[0]

        # 3. 开始倒推回溯
        bestPath = [bestFinalTagId]

        # Python 的 reversed() 可以把一个列表倒过来遍历
        # backpointers 里存的是每一步的脚印列表 bptrsT
        for bptrsT in reversed(backpointers):
            # bptrsT 里面存着从各个标签跳过来的前一步是谁。
            # 现在就站在 bestFinalTagId 上。从 bptrsT 里，
            # 查出到底是谁跳到了 bestFinalTagId 上
            # 查出来之后，把它更新为新的 bestFinalTagId。
            bestFinalTagId = bptrsT[bestFinalTagId]

            # 把它塞进我们要返回的最终路径列表里
            bestPath.append(bestFinalTagId)

        # 4. 翻转与清理
        # 此时的 bestPath 里，还包含着开头的那个 START 标签（因为回溯一直回溯到了起点）
        # 我们把 START 标签弹出来（扔掉），因为我们只需要真正的预测结果。
        start = bestPath.pop()

        # 因为我们是从后往前推的，所以现在的 bestPath 是倒着的（比如 [E, M, B]）
        bestPath.reverse()

        # 返回最终的最高得分，以及这条预测路径
        return pathScore, bestPath
