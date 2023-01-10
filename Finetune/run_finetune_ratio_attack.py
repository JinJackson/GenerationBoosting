from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, AlbertTokenizer, BartTokenizer, BartForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from MatchModel import BertMatchModel, RobertaMatchModel, AlbertMatchModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, DistributedSampler
import os, random
import glob
import torch
import sys
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
import time
import transformers
import re
transformers.logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import accuracy, f1_score, getLogger, TrainData, GenDataset

# sys.path.append('./GenText')

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


logger = None


max_dev_acc = 0
max_dev_f1 = 0
max_test_acc = 0
max_test_f1 = 0


def to_list(x):
    return x.detach().cpu().numpy()

def train(model, tokenizer, checkpoint, attack_method=None, attack_args=None):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    global max_dev_acc
    global max_dev_f1
    global max_test_acc
    global max_test_f1

    train_data = TrainData(data_file=args.train_file,
                           max_length=args.max_length,
                           tokenizer=tokenizer,
                           model_type=args.model_type)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    t_total = len(train_dataloader) * args.epochs

    language = None
    if args.boosting_train:
        if args.dataset == 'LCQMC':
            extra_step_for_one_col = 1000
            language = 'cn'
        elif args.dataset == 'quora':
            extra_step_for_one_col = 2000
            language = 'en'
        else:
            raise Exception

        if args.boosting_col1:
            t_total += extra_step_for_one_col
        if args.boosting_col2:
            t_total += extra_step_for_one_col


    warmup_steps = int(args.warmup_steps * t_total)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # 读取断点 optimizer、scheduler
    checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
            amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataloader))
    logger.debug("  Num Epochs = %d", args.epochs)
    logger.debug("  Batch size = %d", args.batch_size)
    logger.debug("  Total steps = %d", t_total)
    logger.debug("  warmup steps = %d", warmup_steps)
    logger.debug("  Model_type = %s", args.model_type)

    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1

    logger.debug("  Start Batch = %d", checkpoint)
    global_steps = 0
    all_boosting_time = 0
    all_extra_train_time = 0
    all_extra_example = 0
    real_gen_nums = 0
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []

        wrong_case = []

        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", ncols=50):
            model.zero_grad()
            # 设置tensor gpu运行
            query1, query2 = batch[-2:]
            # print(query1, query2)
            batch = tuple(t.to(args.device) for t in batch[:-2])

            if 'roberta' in args.model_type:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels=labels)

            else:
                input_ids, token_type_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels=labels)

            loss = outputs[0]
            
            logits = to_list(outputs[1])
            labels = to_list(labels)

            predict = (logits > 0) + 0
            wrong_case_idx = [idx for idx, res in enumerate((predict ==  labels) == False) if res == True]
            for idx in wrong_case_idx:
                wrong_case.append([query1[idx], query2[idx], str(labels[idx].tolist()[0])])
        

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # 计算出梯度

            epoch_loss.append(loss.item())

            optimizer.step()
            scheduler.step()

            step += 1
            global_steps += 1
            
            if global_steps == warmup_steps:
                wrong_case = []
            
            if global_steps > warmup_steps and step % args.saving_steps == 0:
                logger.debug("loss:"+str(np.array(epoch_loss).mean()))
                logger.debug('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
                dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file, checkpoint=epoch)
                test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file, checkpoint=epoch)
                if dev_acc >= max_dev_acc or dev_f1 >= max_dev_f1 or test_acc >= max_test_acc or test_f1 >= max_test_f1:
                    max_dev_acc = max(max_dev_acc, dev_acc)
                    max_dev_f1 = max(max_dev_f1, dev_f1)
                    max_test_acc = max(test_acc, max_test_acc)
                    max_test_f1 = max(test_f1, max_test_f1)
                    
                    # save checkpoint
                    output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-' + str(step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.debug("Saving model checkpoint to %s", output_dir)
                    # if args.fp16:
                    #     torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.debug("Saving optimizer and scheduler states to %s", output_dir)
                    
                
                # 保存模型
                logger.info(
                    '【DEV】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                        epoch, step, dev_loss, dev_acc, dev_f1))
                logger.info(
                    '【TEST】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                        epoch, step, test_loss, test_acc, test_f1))

                # runing boosting train

                if args.boosting_train:
                    logger.info('Start boosting train for epoch' + str(epoch) + str(step))
                    logger.info('boosting method == ' + attack_method)
                    
                    # writing badcases
                    wrong_case_path = args.save_dir + 'wrong_case/epoch_'+ str(epoch) + '_step' + str(step)
                    if not os.path.exists(wrong_case_path):
                        os.makedirs(wrong_case_path)
                    bad_case_file = wrong_case_path + '/badcases'
                    with open(bad_case_file, 'w', encoding='utf-8') as writer:
                        random.shuffle(wrong_case)
                        # if len(wrong_case) <= args.boarder * args.batch_size:
                        #     boosting_nums = len(wrong_case)
                        # elif int(args.boosting_ratio * len(wrong_case)) <= args.boarder * args.batch_size:
                        #     boosting_nums = args.boarder * args.batch_size
                        # else:
                        #     boosting_nums = args.boarder * args.batch_size
                        boosting_nums = int(args.boosting_ratio * len(wrong_case))

                        wrong_case = wrong_case[:boosting_nums]

                        all_extra_example += (len(wrong_case) * 2)

                        for case in wrong_case:
                            writer.write('\t'.join(case) + '\n')
                    
                    
                    if attack_method == 'Gen':
                        real_gen_nums = all_extra_example
                        para_model = attack_args['para_model']
                        para_tokenizer = attack_args['para_tokenizer']
                        nonpara_model = attack_args['nonpara_model']
                        nonpara_tokenizer = attack_args['nonpara_tokenizer']
                        gen_max_length = attack_args['max_length']
                        gen_language = attack_args['language']


                        col1_case_file = wrong_case_path + '/col1'
                        col2_case_file = wrong_case_path + '/col2'
                    
                        with open(col1_case_file, 'w', encoding='utf-8') as writer1:
                            with open(col2_case_file, 'w', encoding='utf-8') as writer2:
                                for case in wrong_case:
                                    sent1, sent2, label = case
                                    writer1.write(sent1.strip() + '\n')
                                    writer2.write(sent2.strip() + '\n')
                        
            
                        # boosting_train(model, tokenizer, optimizer, scheduler, epoch, step)

                        # device_id = str(args.gen_device)
                        # device_id = '4'
                        
                        source_file = args.save_dir + 'wrong_case/epoch_'+ str(epoch) + '_step' + str(step)

                        start_time = time.time()

                        if args.boosting_col1:
                            # source_file = args.save_dir + 'wrong_case/epoch_'+ str(epoch) + '_step' + str(step) + '/col1'
                            gen_from_file(para_model, para_tokenizer, col1_case_file, gen_language, gen_max_length, 'para')
                            gen_from_file(nonpara_model, nonpara_tokenizer, col1_case_file, gen_language, gen_max_length, 'nonpara')
                            merge_del_repeat = 'sh ./GenText/del_repeat_cat_merge_col1.sh ' + source_file

                            # check_output应该是串行执行，或者Popen() + wait()
                            subprocess.check_output(merge_del_repeat, shell=True)
                        
                        
                        if args.boosting_col2:
                            gen_from_file(para_model, para_tokenizer, col2_case_file, gen_language, gen_max_length, 'para')
                            gen_from_file(nonpara_model, nonpara_tokenizer, col2_case_file, gen_language, gen_max_length, 'nonpara')
                            merge_del_repeat = 'sh ./GenText/del_repeat_cat_merge_col2.sh ' + source_file

                            subprocess.check_output(merge_del_repeat, shell=True)

                        
                        if args.boosting_col1 and args.boosting_col2:
                            if args.boosting_origin:
                                command_merge = 'cat ' + source_file + '/badcases ' + source_file + '/col1_all ' + source_file + '/col2_all > ' + source_file + '/origin_col1_col2_merge_all'
                                subprocess.check_output(command_merge, shell=True)
                                boost_file = source_file + '/origin_col1_col2_merge_all'
                            else:
                                command_merge = 'cat ' + source_file + '/col1_all ' + source_file + '/col2_all > ' + source_file + '/col1_col2_merge_all'
                                subprocess.check_output(command_merge, shell=True)
                                boost_file = source_file + '/col1_col2_merge_all' 
                        
                        elif args.boosting_col1:
                            boost_file = source_file + '/col1_all'
                        
                        elif args.boosting_col2:
                            boost_file = source_file + '/col2_all'

                        end_time = time.time()

                        boosting_time = end_time - start_time

                        all_boosting_time += boosting_time
                        
                    
                    elif attack_method == 'TextAttack':
                        start_time = time.time()
                        attack = attack_args['attack']
                        attackargs = attack_args['attackargs']
                        real_gen_nums = textattack_from_file(attack=attack, attack_args=attackargs, file_path=bad_case_file, extra_nums=real_gen_nums)
                        
                        boost_file = bad_case_file + '_attackfile'

                        end_time = time.time()

                        boosting_time = end_time - start_time
                        
                        all_boosting_time += boosting_time

                    
                    start_train_time = time.time()
                    boost_train_data = TrainData(data_file=boost_file,
                                                max_length=args.max_length,
                                                tokenizer=tokenizer,
                                                model_type=args.model_type)

                    boost_dataLoader = DataLoader(dataset=boost_train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True)

                    # import pdb; pdb.set_trace()
                    logger.debug("***** Running boosting train = %d *****", epoch)
                    logger.debug("  Num examples = %d", len(boost_dataLoader))
                    logger.debug("  Batch size = %d", args.batch_size)

                    

                    model.train()
                    boost_loss = []
                    boost_step = 0

                    # stop_steps = int(args.boosting_ratio * len(boost_dataLoader))

                    for batch in tqdm(boost_dataLoader, desc="boosting training", ncols=50):
                        
                        model.zero_grad()
                        # 设置tensor gpu运行
                        batch = tuple(t.to(args.device) for t in batch[:-2])

                        if 'roberta' in args.model_type:
                            input_ids, attention_mask, labels = batch
                            outputs = model(input_ids=input_ids.long(),
                                            attention_mask=attention_mask.long(),
                                            labels=labels)

                        else:
                            input_ids, token_type_ids, attention_mask, labels = batch
                            outputs = model(input_ids=input_ids.long(),
                                            token_type_ids=token_type_ids.long(),
                                            attention_mask=attention_mask.long(),
                                            labels=labels)

                        loss = outputs[0]
                        
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()  # 计算出梯度

                        boost_loss.append(loss.item())

                        optimizer.step()
                        scheduler.step()
                        
                        boost_step += 1
                    
                    end_train_time = time.time()
                    
                    boosting_train_time = end_train_time - start_train_time
                    
                    all_extra_train_time += boosting_train_time

                    logger.info("loss:"+str(np.array(boost_loss).mean()))
                    logger.info('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
                    dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file, checkpoint=epoch)
                    test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file, checkpoint=epoch)
                    if dev_acc >= max_dev_acc or dev_f1 >= max_dev_f1 or test_acc >= max_test_acc or test_f1 >= max_test_f1:
                        max_dev_acc = max(max_dev_acc, dev_acc)
                        max_dev_f1 = max(max_dev_f1, dev_f1)
                        max_test_acc = max(test_acc, max_test_acc)
                        max_test_f1 = max(test_f1, max_test_f1)
                        output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-' + str(step) + '-boost'
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.debug("Saving model checkpoint to %s", output_dir)
                        # if args.fp16:
                        #     torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.debug("Saving optimizer and scheduler states to %s", output_dir)
                    # 保存模型
                    logger.info(
                        '【Boost DEV】Boost Epoch %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                            epoch, dev_loss, dev_acc, dev_f1))
                    logger.info(
                        '【Boost TEST】Boost Epoch %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                            epoch, test_loss, test_acc, test_f1))
                    
                    wrong_case = []

        
        logger.debug("loss:"+str(np.array(epoch_loss).mean()))
        logger.debug('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file, checkpoint=epoch)
        test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file, checkpoint=epoch)
        if dev_acc >= max_dev_acc or dev_f1 >= max_dev_f1 or test_acc >= max_test_acc or test_f1 >= max_test_f1:
            max_dev_acc = max(max_dev_acc, dev_acc)
            max_dev_f1 = max(max_dev_f1, dev_f1)
            max_test_acc = max(test_acc, max_test_acc)
            max_test_f1 = max(test_f1, max_test_f1)

            output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-' + str(step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (model.module if hasattr(model, "module") else model)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.debug("Saving model checkpoint to %s", output_dir)
            # if args.fp16:
            #     torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.debug("Saving optimizer and scheduler states to %s", output_dir)
            
            
        # 保存模型
        logger.info(
            '【DEV】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                epoch, step, dev_loss, dev_acc, dev_f1))
        logger.info(
            '【TEST】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                epoch, step, test_loss, test_acc, test_f1))
        
    # all_boosting_time = 0
    # all_extra_train_time = 0
    # all_extra_example = 0
    logger.info('【all_boosting_generation_time】：%.4f' % (all_boosting_time))
    logger.info('【all_extra_training_time】：%.4f' % (all_extra_train_time))
    logger.info('【all_extra_example】：%.4f' % (all_extra_example))
    logger.info('【real_gen_example】：%.4f' % (real_gen_nums))
    logger.info('【generation_time_for_each_example】: %.4f' % (all_boosting_time/real_gen_nums))

    logger.info('【BEST TEST ACC】: %.4f,   【BEST TEST F1】: %.4f' % (max_test_acc, max_test_f1))
    logger.info('【BEST DEV ACC】: %.4f,   【BEST DEV F1】: %.4f' % (max_dev_acc, max_dev_f1))


def test(model, tokenizer, test_file, checkpoint, output_dir=None):
    test_data = TrainData(data_file=test_file,
                          max_length=args.max_length,
                          tokenizer=tokenizer,
                          model_type=args.model_type)

    test_dataLoader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    logger.debug("***** Running test {} *****".format(checkpoint))
    logger.debug("  Num examples = %d", len(test_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    loss = []

    all_labels = None
    all_logits = None

    model.eval()

    for batch in tqdm(test_dataLoader, desc="Evaluating", ncols=50):
        with torch.no_grad():
            if 'roberta' in args.model_type:
                batch = [t.to(args.device) for t in batch[:-2]]
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels=labels)

            else:
                batch = [t.to(args.device) for t in batch[:-2]]
                input_ids, token_type_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    acc = accuracy(all_logits, all_labels)
    f1 = f1_score(all_logits, all_labels)

    return np.array(loss).mean(), acc, f1



def boosting_train(model, tokenizer, optimizer, scheduler, epoch, step):
    # sh xx.sh device_id  Gen_type  source_file (3 params)

    global max_dev_acc
    global max_dev_f1
    global max_test_acc
    global max_test_f1
    logger.info('Start boosting train for epoch' + str(epoch) + str(step))
    device_id = str(args.gen_device)

    source_file = args.save_dir + 'wrong_case/epoch_'+ str(epoch) + '_step' + str(step) + '/col1'
    command_para = 'sh ./GenText/generation_from_file.sh '+ device_id + ' ' + 'para' + ' ' + source_file
    command_nonpara = 'sh ./GenText/generation_from_file.sh '+ device_id + ' ' + 'nonpara' + ' ' + source_file
    merge_del_repeat = 'sh ./GenText/del_repeat_cat_merge.sh ' + source_file
    # check_output应该是串行执行，或者Popen() + wait()

    subprocess.check_output(command_para, shell=True)
    subprocess.check_output(command_nonpara, shell=True)
    subprocess.check_output(merge_del_repeat, shell=True)

    
    
    boost_file = source_file + "_all_shuf"
    boost_train_data = TrainData(data_file=boost_file,
                                max_length=args.max_length,
                                tokenizer=tokenizer,
                                model_type=args.model_type)
    boost_dataLoader = DataLoader(dataset=boost_train_data,
                                batch_size=args.batch_size,
                                shuffle=True)
    logger.debug("***** Running boosting train = %d *****", epoch)
    logger.debug("  Num examples = %d", len(boost_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    model.train()
    boost_loss = []
    for batch in tqdm(boost_dataLoader, desc="boosting training", ncols=50):
        model.zero_grad()
        # 设置tensor gpu运行
        batch = tuple(t.to(args.device) for t in batch[:-2])

        if 'roberta' in args.model_type:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels)

        else:
            input_ids, token_type_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels)

        loss = outputs[0]
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()  # 计算出梯度

        boost_loss.append(loss.item())

        optimizer.step()
        scheduler.step()


    logger.info("loss:"+str(np.array(boost_loss).mean()))
    logger.info('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
    dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file, checkpoint=epoch)
    test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file, checkpoint=epoch)
    if dev_acc >= max_dev_acc or dev_f1 >= max_dev_f1 or test_acc >= max_test_acc or test_f1 >= max_test_f1:
        max_dev_acc = max(max_dev_acc, dev_acc)
        max_dev_f1 = max(max_dev_f1, dev_f1)
        max_test_acc = max(test_acc, max_test_acc)
        max_test_f1 = max(test_f1, max_test_f1)
    # 保存模型
    logger.info(
        '【Boost DEV】Boost Epoch %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
            epoch, dev_loss, dev_acc, dev_f1))
    logger.info(
        '【Boost TEST】Boost Epoch %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
            epoch, test_loss, test_acc, test_f1))
    output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-boost'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.debug("Saving model checkpoint to %s", output_dir)
    # if args.fp16:
    #     torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.debug("Saving optimizer and scheduler states to %s", output_dir)
    

def gen_from_file(model, tokenizer, file_path, language, max_length, gen_type):
    def load_and_cache_examples(tokenizer, file_path):
        dataset = GenDataset(tokenizer, data_dir=file_path, max_source_length=256)
        return dataset
    if language == 'cn':
        batch_size = 16
        flag = True
        sep_sign = ''
    elif language == 'en':
        batch_size = 8
        flag = False
        sep_sign = ' '
    else:
        assert False, 'language should be in [cn, en]'
    eval_dataset = load_and_cache_examples(tokenizer, file_path)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)

    # Gen!!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)


    preds = []

    decoder_start_token_id = tokenizer.bos_token_id
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            with torch.no_grad():
                generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=5, 
                    length_penalty=0.6, max_length=max_length, repetition_penalty=2.0, decoder_start_token_id=decoder_start_token_id)
                gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if flag:
                    gen_text = [t if (t.endswith('？') or t.endswith('?')) else t+' ？' for t in gen_text]
                preds += gen_text
    # written_file = args.data_dir + '_generated'
    original_data = []
    with open(file_path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        original_data = [line.strip() for line in lines]
    
    if gen_type not in ['para', 'nonpara']:
        print('gen_type wrong, plz input \"para\" or \"nonpara\"')
    
    label = 1 if gen_type == 'para' else 0
    print(len(original_data), len(preds))
    assert len(original_data) == len(preds)

    written_file = file_path + "_" + gen_type + "_generated"
    
    with open(written_file, "w", encoding='utf-8') as writer:
        for origin, gen in zip(original_data, preds):
            gen = sep_sign.join(gen.split())
            writer.write(origin + '\t' + gen + '\t' + str(label) + '\n')
    


def textattack_from_file(attack, attack_args, file_path, extra_nums):
    #data = [(("A man is sleeping on the bed.", "The man is almost sleeping."), 1), (("The man is almost sleeping.", "A man is sleeping on the bed."), 0)]
    with open(file_path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
    raw_data = [line.split('\t') for line in lines]
    data = []
    for a_data in raw_data:
        data.append(((a_data[0], a_data[1]), int(a_data[2])))
        data.append(((a_data[1], a_data[0]), int(a_data[2])))
    dataset = textattack.datasets.Dataset(data, input_columns=['text1', 'text2'])
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attack_res = attacker.attack_dataset()
    # for result in attack_res:
    print('attack_res length == data length?', len(attack_res) == len(dataset))
    attack_data = []
    tmp_nums = extra_nums
    for perturbed_res, original in zip(attack_res, data):
        if not isinstance(perturbed_res, SkippedAttackResult):
            tmp_nums += 1
        try:
            perturbed_text1, perturbed_text2 = [t[7:] for t in perturbed_res.perturbed_text().split('\n')]
        except:
            re_tool = re.compile(r"<S.*>")
            perturbed_text = re_tool.sub('\n', perturbed_res.perturbed_text())
            perturbed_text1, perturbed_text2 = [t for t in perturbed_text.split('\n')]

        label = str(original[1])
        attack_data.append([perturbed_text1, perturbed_text2, label])
    written_file = file_path + '_attackfile'
    with open(written_file, 'w', encoding='utf-8') as writer:
        for data in attack_data:
            writer.write('\t'.join(data) + '\n')
    return tmp_nums
        


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/LCQMC/clean/train_clean.txt')
    parser.add_argument('--dev_file', default='data/LCQMC/clean/dev_clean.txt')
    parser.add_argument('--test_file', default='data/LCQMC/clean/test_clean.txt')

    parser.add_argument('--model_type', default='bert-base-chinese')
    parser.add_argument('--seed', default=2048, type=int)
    parser.add_argument('--save_dir', default='result/BQ/bert/VAE/checkpoints')
    parser.add_argument('--do_train', default=True)
    parser.add_argument('--do_lower_case', default=True)


    # TODO  常改参数

    #超参数
    parser.add_argument('--learning_rate', default='1e-5', type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--warmup_steps', default=0.1, type=float)
    parser.add_argument('--saving_steps', default=1000, type=int)
    parser.add_argument('--gen_device', default=1000, type=int)
    parser.add_argument('--boosting_train', action='store_true')
    parser.add_argument('--boosting_col1', action='store_true')
    parser.add_argument('--boosting_col2', action='store_true')
    parser.add_argument('--boosting_origin', action='store_true')
    parser.add_argument('--boosting_ratio', default=1.0, type=float)
    parser.add_argument('--boosting_method', default='Gen', type=str)
    parser.add_argument('--boarder', default=10, type=int)
    parser.add_argument('--dataset', required=True, type=str)
    

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fptype', default='O2')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    if args.seed > -1:
        seed_torch(args.seed)

    # 创建存储目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = getLogger(__name__, os.path.join(args.save_dir, 'log.txt'))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        else:
            amp = None

    if 'roberta' in args.model_type:
        MatchModel = RobertaMatchModel
        Tokenizer = RobertaTokenizer
    elif 'albert' in args.model_type:
        MatchModel = AlbertMatchModel
        Tokenizer = AlbertTokenizer
    elif 'bert' or 'ernie' in args.model_type:
        MatchModel = BertMatchModel
        Tokenizer = BertTokenizer

    if args.do_train:
        # train： 接着未训练完checkpoint继续训练
        checkpoint = -1
        for checkpoint_dir_name in glob.glob(args.save_dir + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
            except Exception as e:
                pass
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
        if checkpoint > -1:
            logger.debug(f" Load Model from {checkpoint_dir}")

        tokenizer = Tokenizer.from_pretrained(args.model_type if checkpoint == -1 else checkpoint_dir,
                                              do_lower_case=args.do_lower_case)
        model = MatchModel.from_pretrained(args.model_type if checkpoint == -1 else checkpoint_dir)
        model.to(args.device)


        # 训练
        if args.boosting_train:
            if args.boosting_method == 'Gen':
                model_class = BartForConditionalGeneration
                if args.dataset == 'LCQMC':
                    gen_language = 'cn'
                    gen_max_length = 32
                    tokenizer_class = BertTokenizer
                    nonpara_model_path=r"/data/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/LCQMC/midones/nonpara/finetuned-bart-large-chinese/bs8_accumulation4_epoch3_lr2e-5_seed15213"
                    para_model_path=r"/data/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/LCQMC/midones/para/finetuned-bart-large-chinese/bs8_accumulation4_epoch3_lr2e-5_seed15213"
                elif args.dataset == 'quora':
                    gen_language = 'en'
                    gen_max_length = 128
                    tokenizer_class = BartTokenizer
                    nonpara_model_path=r"/data/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/quora2/nonpara/finetuned-bart-large/bs8_accumulation2_epoch3_lr2e-5_seed15213"
                    para_model_path=r"/data/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/quora2/para/finetuned-bart-large/bs8_accumulation2_epoch3_lr2e-5_seed15213"
                else:
                    assert False, 'dataset should be in [LCQMC, quora]'
                
                para_model = model_class.from_pretrained(para_model_path).to(args.device)
                nonpara_model = model_class.from_pretrained(nonpara_model_path).to(args.device)
                para_tokenizer = tokenizer_class.from_pretrained(para_model_path)
                nonpara_tokenizer = tokenizer_class.from_pretrained(nonpara_model_path)

                attack_args = {
                    "para_model": para_model,
                    "nonpara_model": nonpara_model,
                    "para_tokenizer": para_tokenizer,
                    "nonpara_tokenizer": nonpara_tokenizer,
                    "max_length": gen_max_length,
                    "language": gen_language
                }
                train(model, tokenizer, checkpoint, attack_method='Gen', attack_args=attack_args)
            elif args.boosting_method == 'TextAttack':
                logger.debug("loading textattack")
                import textattack
                from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
                logger.debug("loading attack model")
                attack_model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP')
                attack_tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')
                attack_model.to(args.device)
                logger.debug("wrappering")
                model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(attack_model, attack_tokenizer)
                logger.debug("loading recipes and attack args")
                attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)
                #TextBuggerLi2018
                #PWWSRen2019
                #BERTAttackLi2020
                #CLARE2020
                #BAEGarg2019
                #BERTAttackLi2020
                attackargs = textattack.AttackArgs(num_examples=-1, random_seed=765, checkpoint_interval=None, disable_stdout=True)
                attack_args = {
                    'attack': attack,
                    'attackargs':attackargs
                }
                logger.debug('loading textattack done, start training')
                train(model, tokenizer, checkpoint, attack_method="TextAttack", attack_args=attack_args)


                # from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
                
            else:
                assert False, 'boosting method should be in [Gen, TextAttack]'

        else:
            train(model, tokenizer, checkpoint, attack_method=None, attack_args=None)

    else:
        # eval：指定模型
        checkpoint = args.checkpoint
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
        tokenizer = Tokenizer.from_pretrained(checkpoint_dir, do_lower_case=args.do_lower_case)
        model = MatchModel.from_pretrained(checkpoint_dir)
        model.to(args.device)
        # 评估
        test_loss, test_acc, test_f1 = test(model, tokenizer, test_file=args.test_file, checkpoint=checkpoint)
        logger.debug('Evaluate Epoch %d: test_loss=%.4f, test_acc=%.4f, test_f1=%.4f' % (checkpoint, test_loss, test_acc, test_f1))