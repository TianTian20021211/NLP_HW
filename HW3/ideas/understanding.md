对于需要做的事情的理解，不是计划

# 数据部分

给的是接近 2w 个公司的 earning calls 的文本提取出来的数据

每个公司在给的数据的这段时间中都有很多 earning calls

每个 earning call 被用 signal 指标切分成很多不同的部分，主要是按照谁在发言，也有总共的内容

有一个公司对这些 earning calls 做了 nlp 处理，得到了一些分析结果

对于一个公司的一次 earning call 中的 signal 指标切分出来的一个部分，有以下这些具体的 nlp 结果

- eventScore 类别：出现的 event 在某一方面的得分的整合
  - 例子：EventPos_1_1_1意思是，这段话的 event 在 1_1_1 版本的正面情绪衡量中的得分
  - 总共有四个不同的版本，4_2_1 是官方真正使用过的版本
  - 有 Pos, Neg, Score 三种不同的得分，Score 就是综合的得分
- ATCClassifierScore 是一个很重要的综合分数，综合了提供的所有这些指标
- AspectTheme系列是四个类别的组合，数值代表这段话中有多少句话属于这个类别
  - 例子：AspectTheme_Forecast_FinancialPerformance_High_Negative
  - Aspect：这句话表明了什么类型的内容，Forecast（这个不是随便弄的分类，有很具体的背后含义）
  - Theme：某一个特定的商业主题，FinancialPerformance
  - Magnitude：情绪的强度，High
  - Sentiment：情绪方向，Negative

# 要做什么

在 S&P 500, S&P 1500, Russell 3000 这三个 universe 上做策略 & 搭回测框架

在 daily / weekly / monthly 这三个不同频率的 rebalance 下弄，选一个表现最好的

