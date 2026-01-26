
from transformers import BertTokenizer
import pickle # 保存pkl
from tqdm import tqdm
import os

def data_preprocess(train_txt_path, train_pkl_path):
    """
    [CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]
    :param train_txt_path:
    :param train_pkl_path:
    :return:
    """
    tokenizer = BertTokenizer(
        '../vocab/vocab.txt',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )
    # 查看词表
    # print(f'tokenizer.vocab_size-->{tokenizer.vocab_size}')
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    # sep_id -->102, cls_id -->101
    # print(f'sep_id-->{sep_id}, cls_id-->{cls_id}')

    # 读取train数据
    with open(train_txt_path, 'rb') as f:
        data = f.read().decode('utf-8')
    # print(f'data-->{data}')

    # win和linux换行符不同
    if '\r\n' in data:
        train_data = data.split('\r\n\r\n')
    else:
        train_data = data.split('\n\n')

    # tokenize之后的长度
    dialogue_len = []
    # 记录所有对话
    dialogue_list = []

    for index, dialogue in enumerate(tqdm(train_data)):
        # print(f'index-->{index}, dialogue-->{dialogue}')
        if '\r\n' in dialogue:
            sequences = dialogue.split('\r\n')
        else:
            sequences = dialogue.split('\n')
        # print(f'sequences-->{sequences}')
        input_ids = [cls_id]
        for sequence in sequences:
            # print(f'sequence-->{sequence}')
            # print(f'tokenizer.encode(sequence)-->{tokenizer.encode(sequence)}')
            # print(f'tokenizer.encode(sequence, add_special_tokens=False)-->{tokenizer.encode(sequence, add_special_tokens=False)}')
            # input_ids.append(tokenizer.encode(sequence, add_special_tokens=False)) # 这里用append加了个新列表
            input_ids += tokenizer.encode(sequence, add_special_tokens=False)
            input_ids.append(sep_id)
        print(f'input_ids-->{input_ids}')

        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)
    print(f'dialogue_len-->{dialogue_len}')
    print(f'dialogue_list-->{dialogue_list[:2]}')
    # 保存数据
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(dialogue_list, f)






if __name__ == '__main__':
    train_txt_path = '../data/medical_valid.txt'
    train_pkl_path = '../data/medical_valid1.pkl'
    data_preprocess(train_txt_path, train_pkl_path)