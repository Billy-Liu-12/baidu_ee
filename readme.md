任务：抽取句子中包含的事件类型和当前类型对应的论元角色集合。
    训练数据给出了触发词-事件类型-论元角色
    预测结果不包括触发词，需要事件列表（类型）以及对应的论元集合以及对应的角色。

```python
### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```


SBV  --主语
VOB --宾语
IOB 
FOB
COO 并列（共享主宾，共享谓语）

move = {0:0,1:66,2:100,3:132,4:170,5:238,6:288,7:360,8:406}
class_tag_num = {0: 67, 1: 35, 2: 33, 3: 39, 4: 69, 5: 51, 6: 73, 7: 47, 8: 29}

0. 财经/交易:1150   , 论元个数：33      ,tag_num:67
1. 产品行为:2000    , 论元个数：17      ,tag_num:35
2. 交往:450        , 论元个数：16      ,tag_num:33
3. 竞赛行为:2770    , 论元个数：19      ,tag_num:39   效果最差
4. 人生:2330       , 论元个数：34      ,tag_num:69   差
5. 司法行为:1610    , 论元个数：25      ,tag_num:51
6. 灾害/意外:970    , 论元个数：36      ,tag_num:73
7. 组织关系:1430    , 论元个数：23      ,tag_num:47   差
8. 组织行为:430     , 论元个数：14      ,tag_num:29



### bert+crf
如果序列标注的种类过多，crf的学习率是需要给较大的一个学习率的。--从tag loss 和crf loss的对比上可以看出这种现象，crf的loss 是tag loss的一百多倍。

### 加入分词，pos，ner 的bert与原文本对齐问题
1. 从文本id 映射到-> 分词后的词的id:   [1,1,2,2,2,3,3]-->[1,2,3]                     
2. 词的id 映射到-> bert 分词后的范围(start,end)  [1,2,3,4]-->[(1,4),(4,5),(5,8),(8,11)] 
### 加入依存句法分析
用邻接矩阵来表示依存句法关系，在forward中gcn

### 使用RNN中的DataParallel问题
1. RuntimeError: Gather got an input of invalid size: got [32, 7, 15013], but expected [32, 6, 15013] (gather at /pytorch/torch/csrc/cuda/comm.cpp:239
该模型被放到不同的GPU上，而模型返回的数据在合并gather的时候，出现了维度不同无法gather的问题
出错原因：
错误发生在model代码内使用torch.nn.utils.rnn.pad_packed_sequence() 工具时，在将数据分配到不同的GPU上时，每个GPU的该函数都会自身执行，其会取它所看到的所有sequence的最长长度，来对其他非最长的句子进行padding。这时每个GPU上的数据不同会导致每个函数看到的最长长度不同，导致padding后的长度不同，这样在gather的时候，就会出现维度不匹配的问题。
解决方法：
使用pad_packed_sequence()的total_length参数来确保forward()调用相同长度的返回序列
```python
total_length = padded_input.size(1)  # get the max sequence length
output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
```
上述代码会存在UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greately increasing memory usage. To compact weights again call flatten_parameters()
可以通过在forward 函数中添加 self.rnn.flatten_parameters()


## tricks
### 损失截断--防止过拟合
### 调整类别损失
### 主动限制crf的转移矩阵


### 猜想会遇到的问题，但是没有遇到：句子长度超过512.