import os
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch.nn.functional as F
from parameter_config import *

PAD = '[PAD]'
pad_id = 0

# 获取项目根目录（脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pconf = ParameterConfig()
# 当用户使用GPU,并且GPU可用时
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:{}'.format(device))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 使用绝对路径
vocab_path = os.path.join(BASE_DIR, pconf.vocab_path.lstrip('./'))
model_path = os.path.join(BASE_DIR, 'save_model/epoch97')

tokenizer = BertTokenizerFast(
    vocab_file=vocab_path,
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]")
# model = GPT2LMHeadModel.from_pretrained('./save_model/epoch97')
model = GPT2LMHeadModel.from_pretrained(model_path) # epoch97是闲聊，savemodel2未训练完全
model = model.to(device)
model.eval()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    结合 Top-K 和 Top-P 的过滤函数
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))

    # 1. Top-K 过滤
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

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

def model_predict(text):
    history = []
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    history.append(text_ids)
    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
    for history_id, history_utr in enumerate(history[-pconf.max_history_len:]):
        input_ids.extend(history_utr)
        input_ids.append(tokenizer.sep_token_id)
    input_ids = torch.tensor(input_ids).long().to(device)
    input_ids = input_ids.unsqueeze(0)
    response = []  # 根据context，生成的response
    for _ in range(pconf.max_len):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        for id in set(response):
            next_token_logits[id] /= pconf.repetition_penalty
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=pconf.topk, top_p=pconf.topp)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        response.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
    history.append(response)
    text = tokenizer.convert_ids_to_tokens(response)
    return "".join(text)

