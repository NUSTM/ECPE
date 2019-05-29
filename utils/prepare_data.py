# encoding:utf-8

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time

def print_time():
    print '\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend( [emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # 每个词及词的位置
    
    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    doc_id = []
    
    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)
        y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            y_po[i][int(i+1 in pos)]=1
            y_ca[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        
        y_position.append(y_po)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    
    y_position, y_cause, x, sen_len, doc_len = map(np.array, [y_position, y_cause, x, sen_len, doc_len])
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_position, y_cause, y_pairs, x, sen_len, doc_len

def load_data_2nd_step(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []
    
    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval(inputFile.readline().strip())
        pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])
        sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        pos_list, cause_list = [], []
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            if int(line[1].strip())>0:
                pos_list.append(i+1)
            if int(line[2].strip())>0:
                cause_list.append(i+1)
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = doc_id*10000+i*100+j
                pair_id.append(pair_id_cur)
                y.append([0,1] if pair_id_cur in pair_id_all else [1,0])
                x.append([x_tmp[i-1],x_tmp[j-1]])
                sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                distance.append(j-i+100)
    y, x, sen_len, distance = map(np.array, [y, x, sen_len, distance])
    for var in ['y', 'x', 'sen_len', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
    print('load data done!\n')
    return pair_id_all, pair_id, y, x, sen_len, distance

def acc_prf(pred_y, true_y, doc_len, average='binary'): 
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def prf_2nd_step(pair_id_all, pair_id, pred_y, fold = 0, save_dir = ''):
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir+'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0]/10000, pair_id_all[-1]/10000
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e+1):
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                true_pair.append((p1, p2))
                line += '({}, {}) '.format(p1,p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all): break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair)>1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)
    if fold:
        write_log()
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)
    o_acc_num = len(s1 & s2)
    acc_num = len(s1 & s3)
    o_p, o_r = o_acc_num/(len(s2)+1e-8), o_acc_num/(len(s1)+1e-8)
    p, r = acc_num/(len(s3)+1e-8), acc_num/(len(s1)+1e-8)
    f1, o_f1 = 2*p*r/(p+r+1e-8), 2*o_p*o_r/(o_p+o_r+1e-8)
    
    return p, r, f1, o_p, o_r, o_f1, keep_rate
    
