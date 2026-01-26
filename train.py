import torch
import torch.optim as optim
import os
from datetime import datetime
import transformers
# GPT2
from transformers import GPT2LMHeadModel, GPT2Config
# BERT分词器
from transformers import BertTokenizerFast
from unstructured.metrics.text_extraction import calculate_accuracy

# 损失，准确率
from function_tools import *
# 项目配置
from parameter_config import *
# 数据导入
from data_preprocess.dataloader import *


def train_epoch(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        epoch,
        args,
):
    """

    :param model:GPT2
    :param train_dataloader:训练数据集
    :param optimizer:优化器
    :param scheduler:学习率预热
    :param epoch:当前轮次
    :param args:模型配置文件的参数对象
    :return:
    """
    model.train()
    device = args.device
    # ignore_index label token不计算梯度
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0 # 整个epoch的loss

    # 预测对的，预测word总量
    epoch_correct_num, epoch_total_num = 0, 0

    # batch
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        # 如果输入时候不仅只有inputs还有label，所以结果直接有loss
        outputs = model.forward(input_ids, labels=labels)
        # 如果对模型的输入只有input，那么模型的结果不会含有loss值，此时，可以自定义函数来计算损失
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()
        # 统计这个batch预测token正确数和总数
        batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)

        # 这个batch的accuracy
        batch_acc = batch_correct_num / batch_total_num
        # 统计该epoch的预测token的正确数与总数
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num

        total_loss += loss.item()
        # self.gradient_accumulation_steps = 4， 累积的步数
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        # 梯度裁剪 # 避免梯度爆炸的方式。self.max_grad_norm = 2.0 # l2范数
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # 进行一定step的梯度累计之后，更新参数
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0: # 梯度累积步数
            optimizer.step() # 更新参数
            scheduler.step() # 更新学习率
            optimizer.zero_grad() # 梯度清零

        if (batch_idx + 1) % args.loss_step == 0:
            print(
                f"batch {batch_idx + 1} of epoch {epoch + 1},loss {loss.item() * args.gradient_accumulation_steps},batch_acc {batch_acc}, lr {scheduler.get_lr()}"
            )
        del input_ids, outputs
        break

    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print(
        f"epoch {epoch + 1}: loss {epoch_mean_loss}, predict_acc {epoch_mean_acc}"
    )
    # save
    if epoch % 10 == 0 or epoch == args.epochs:
        print(f'saving model for {epoch + 1}')
        model_path = os.path.join(args.save_model_path, f'bj_epoch{epoch + 1}')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # 保存预训练模型的方式
        model.save_pretrained(model_path)
        print(f'epoch {epoch + 1} finished')
        epoch_finish_time = datetime.now()
        print(f'time for one epoch: {epoch_finish_time - epoch_start_time}')
    return epoch_mean_loss

def validate_epoch(model, validate_dataloader, epoch, args):
    print('start validating')
    model.eval()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)

            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            total_loss += loss.item()
            del input_ids, outputs
            break
        # 记录当前epoch的平均loss
        epoch_mean_loss = total_loss / len(validate_dataloader)
        print(
            f"epoch {epoch + 1}: loss {epoch_mean_loss}"
        )
        epoch_finish_time = datetime.now()
        print(f'time for one epoch: {epoch_finish_time - epoch_start_time}')
        return epoch_mean_loss


def train(model, train_dataloader, validate_dataloader, args):
    # t_total模型训练完毕，一共要迭代多少步
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    """
    这里对于模型的参数，分别进行权重参数的衰减优化：防止过拟合，以及学习率预热处理优化：
    在初始阶段将学习率从较小的值逐步增加到设定的初始值，然后按照设定的学习率调整策略进行训练。
    学习率预热的目的是让模型在初始阶段更快地适应数据，避免训练过程中因为学习率过大导致的梯度爆炸等问题，
    从而提高模型的训练效果和泛化性能。
    optimizer： 优化器
        这个参数需要传入一个优化器对象（optimizer object）。它代表在训练过程中用于更新模型参数的优化器，比如Adam或SGD等。
    num_warmup_steps：初始预热步数
        这个参数确定学习率在开始阶段从0线性增加到初始值的步数。在Transformer模型中，通过逐渐增加学习率来稳定和加速训练过程是常见的做法。通常，这个值是总训练步数的一小部分。
    num_training_steps：整个训练过程的总步数
        这个参数指定了总的训练步数或迭代次数。它表示优化器将在给定数据集上进行多少次参数更新。
    """
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    print('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(args.epochs):
        # train
        train_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
        )
        train_losses.append(train_loss)
        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model,
            validate_dataloader=validate_dataloader,
            epoch=epoch,
            args=args,
        )
        validate_losses.append(validate_loss)

        # 保存当前损失最低的模型，损失低
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            print(f'saving current best model for epoch {epoch + 1}')
            model_path = os.path.join(args.save_model_path, 'min_ppl_model_bj'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model.save_pretrained(model_path)
        break

def main():
    # 初始化配置参数
    params = ParameterConfig()
    # 设置使用哪些显卡进行训练:默认为0
    # 如果你的电脑有大于1张的显卡，可以选择使用
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'数字0代表你的第一张显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'数字1代表你的第二张显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] ='0, 1'代表同时利用0和1两张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(
        params.vocab_path,
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
    )
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    # 创建模型的输出目录
    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)
    # 创建模型
    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        model_config = GPT2Config.from_pretrained(params.config_json)
        print(model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(params.device)
    print(f'model.config.vocab_size-->{model.config.vocab_size}')
    print(f'tokenizer.vocab_size-->{tokenizer.vocab_size}')
    # 确认
    assert model.config.vocab_size == tokenizer.vocab_size
    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(f'模型参数总量：{num_parameters}')
    # 加载训练集和验证集
    train_dataloader, validate_dataloader = get_dataloader(params.valid_path,params.valid_path)
    train(model, train_dataloader, validate_dataloader, params)

if __name__ == '__main__':
    main()






