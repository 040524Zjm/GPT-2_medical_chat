import os
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
from parameter_config import *

PAD = '[PAD]'
pad_id = 0
# top-k底层复制来
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    使用top-k和/或nucleus（top-p）筛选来过滤logits的分布
        参数:
            logits: logits的分布，形状为（词汇大小）
            top_k > 0: 保留概率最高的top k个标记（top-k筛选）。）。

    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check：确保top_k不超过logits的最后一个维度大小

    if top_k > 0:
        # 移除概率小于top-k中的最后一个标记的所有标记
        # torch.topk()返回最后一维中最大的top_k个元素，返回值为二维(values, indices)
        # ...表示其他维度由计算机自行推断
        # print(f'torch.topk(logits, top_k)--->{torch.topk(logits, top_k)}')
        # print(f'torch.topk(logits, top_k)[0]-->{torch.topk(logits, top_k)[0]}')
        # print(f'torch.topk(logits, top_k)[0][..., -1, None]-->{torch.topk(logits, top_k)[0][..., -1, None]}')
        # print(f'torch.topk(logits, top_k)[0][-1]-->{torch.topk(logits, top_k)[0][-1]}')
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # print(f'indices_to_remove--->{indices_to_remove}')
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
        # print(f'logits--->{logits}')
    # 2. Top-P (Nucleus) 过滤
    if top_p > 0.0:
        # 对 logits 进行降序排列
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算 Softmax 后的累积概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过 top_p 的标记（保留第一个超过阈值的词）
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将掩码向右移动，以确保保留第一个超过 top_p 的词
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # 排序的掩码后的要改回原来的索引
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    pconf = ParameterConfig()
    # 当用户使用GPU,并且GPU可用时
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device--->{device}')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tokenizer = BertTokenizer(
        vocab_file=pconf.vocab_path,
        sep_token='[SEP]',
        cls_token='[CLS]',
        pad_token='[PAD]',
    )
    model = GPT2LMHeadModel.from_pretrained('./save_model/epoch97') # save_model1
    model = model.to(device)
    model.eval()
    # 历史
    history = []
    print('欢迎使用GPT2-Medical-Chatbot，输入“quit”或者“exit”可以退出！')
    while True:
        try:
            text = input('用户：')
            if text == 'quit' or text == 'exit':
                print('Bye-bye')
                break
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            # 历史
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]
            for history_id, history_utr in enumerate(history[-pconf.max_history_len:]):
                input_ids.extend(history_utr) # extend()多个值（用新列表扩展原来的列表）。
                input_ids.append(tokenizer.sep_token_id)
                # 历史记录 + sep + 当前输入
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            input_ids = input_ids.unsqueeze(0)
            response = [] # 由context生成的回复
            # 最多生成长度
            for _ in range(pconf.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :] # 生成下一个单词的概率值

                # todo 已生成的结果generated中的每个token添加一个 重复惩罚项，降低其生成概率
                for id in set(response):# 去重后还在。
                    next_token_logits[id] /= pconf.repetition_penalty # 重复惩罚
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                # top-k top-p
                filter_logits = top_k_top_p_filtering(next_token_logits, top_k=pconf.topk, top_p=pconf.topp) # 发散生成

                # torch.multinomial表示表示从候选集合中无放回的抽取num sample个，权重越高几率高，返回下标
                next_token = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id: # 遇到sep停止 每生成一个 Token，就拼接到输入序列末尾，模型基于 “原序列 + 新 Token” 预测下一个 Token，直到达到 max_len 或遇到 [SEP]；
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print('Chatbot: ' + ''.join(text))
        except KeyboardInterrupt:
            break



if __name__ == '__main__':
    main()
