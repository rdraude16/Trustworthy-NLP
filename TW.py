import string
import re
import csv
from os import listdir
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot
from numpy import array
import numpy as np
from random import sample
import pandas as pd
import statistics

###Global Variables:
EXPECTED = 'Expected_Data.csv'
CONSISTENT = 'Consistent_Data.csv'
UNCERTAIN = 'Uncertain_Data.csv'
METHODS = 'Methods_Data.csv'
ETHICS = 'Ethics_Data.csv'
PEER = 'Peer_Data.csv'
STATS = 'Stats_Data.csv'



#### Data Preparation
####
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

vocabulary_filename = 'vocab_train_trustworthy_full_Apr2022.txt'
fir = load_doc(vocabulary_filename)
sec = fir.split()
VOCABULARY = set(sec)
conv_vocab_name = 'conv_vocab'
o = load_doc(conv_vocab_name)
t = o.split()
CONV_VOCAB = set(t)
u = load_doc('bigram_vocab')
h = u.splitlines()
BIGRAM_VOCAB = set(h)



def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # convert to lower case
    tokens = [word.lower() for word in tokens]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    #print(type(tokens))
    return tokens


# load doc, clean and return line of tokens
def doc_to_line(doc, vocab):
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    #Bag of bigrams only:
    tokens = zip(tokens,tokens[1:])
    bitokens=[]
    for tuple in tokens:
        collapsed = ' '.join(tuple)
        bitokens.append(collapsed)
        #print(bitokens)
    #Bag of words/bigrams only:
    tokens = [w for w in bitokens if w in vocab]

    return ' '.join(tokens)

def process_docs(directory, vocab, is_train, fold_num):
    lines = list()
    labels = list()
    if directory == 'Expected_Data.csv':
        code_column_index= 'Expected Result'
    elif directory == 'Consistent_Data.csv':
        code_column_index='Consistent Results'
    elif directory == 'Uncertain_Data.csv':
        code_column_index='Uncertainty'
    elif directory == 'Methods_Data.csv':
        code_column_index='Good Methods'
    elif directory == 'Peer_Data.csv':
        code_column_index='Peer Review'
    elif directory == 'Stats_Data.csv':
        code_column_index='Statistics'
    elif directory == 'Ethics_Data.csv':
        code_column_index='Ethics'

    df = pd.read_csv(directory)
    fifth = len(df.index)/5
    df = df.T
    line_count = 0
    for row in df:
        if line_count == 0:
            line_count += 1
            continue
        if df.loc[code_column_index,row] =="":
            line_count += 1
            continue

        ### Handling different folds for train/test

        if fold_num == 1:
            if is_train and line_count<=fifth:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
            elif is_train:
                line_count += 1
                continue
            elif line_count>=fifth:
                line_count += 1
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
            else:
                line_count += 1
                continue
        if fold_num == 2:
            if is_train and fifth<line_count<=2*fifth:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
            elif is_train:
                line_count += 1
                continue
            elif fifth<line_count<=2*fifth:
                line_count += 1
                continue
            else:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
        if fold_num == 3:
            if is_train and 2*fifth<line_count<=3*fifth:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                line_count += 1
                labels.append(label)
            elif is_train:
                line_count += 1
                continue
            elif 2*fifth<line_count<=3*fifth:
                line_count += 1
                continue
            else:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
        if fold_num == 4:
            if is_train and 3*fifth<line_count<=4*fifth:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
            elif is_train:
                line_count += 1
                continue
            elif 3*fifth<line_count<=4*fifth:
                line_count += 1
                continue
            else:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
        if fold_num == 5:
            if is_train and 4*fifth<line_count:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1
            elif is_train:
                line_count += 1
                continue
            elif 4*fifth<line_count:
                line_count += 1
                continue
            else:
                line = doc_to_line(df.loc['Trustworthy Response',row],vocab)
                lines.append(line)
                label = int(df.loc[code_column_index,row])
                labels.append(label)
                line_count += 1

    labels = array(labels)

    #I deleted the sample stuff here because it is already in a random order

    print(len(lines))
    print("FREQUENCY:")
    print(sum(labels))
    return (lines, labels)
#load and clean a dataset
def load_clean_dataset(vocab, is_train, fold_num, directory):
    #load documents
    docs, labels = process_docs(directory, vocab, is_train,fold_num)
    return docs, labels

def define_model(n_words, vocab_size, input_length):
    #CNN or Embedding: vocab_size, input_length in arguments
    #Embedding (CNN or not): matrix in arguments
    # define network
    model = Sequential()
    #Bag of Words only:
    # model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))

    #Regular embedding only:
    # e = Embedding(vocab_size, 100, weights=[matrix], input_length=input_length, trainable=False)
    # model.add(e)
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))

    #CNN only:

    #vocab:
    model.add(Embedding(vocab_size, 100, input_length=input_length))
    #embedding:
    # e = Embedding(vocab_size, 100, weights=[matrix], input_length=input_length, trainable=False)

    model.add(Conv1D(32, 8, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #All:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','TruePositives','FalsePositives','FalseNegatives','TrueNegatives'])
    return model

# evaluate a neural network model
def bows_stats(vocab,fold_num, directory):
    scores = list()
    tps = list()
    fps = list()
    fns = list()
    tns = list()
    freq = list()
    n_training_repeats = 1
    for i in range(n_training_repeats):
        train_docs, ytrain = load_clean_dataset(vocab, True, fold_num, directory)
        test_docs, ytest = load_clean_dataset(vocab, False, fold_num, directory)
        Xtrain, Xtest, tokenizer = prepare_data(train_docs, test_docs, 'binary')
        n_repeats = 10
        print(Xtest.shape)
        n_words = Xtest.shape[1]
        freq.append(sum(ytrain))
        for i in range(n_repeats):
            # define network
            model = define_model(n_words)
            # fit network
            model.fit(Xtrain, ytrain, epochs=15, verbose=0)
            # evaluate
            dat, acc, tp, fp, fn, tn = model.evaluate(Xtest, ytest, verbose=0)
            print(dat)
            scores.append(acc)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
            print('%d accuracy: %s' % ((i+1), acc))
    return scores, tps, fps, fns, tns, (freq,len(ytrain))


# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode, input_length):
    # CNN or Embedding: input_length in arguments
    # Embedding (CNN or not): matrix, vocab_size in arguments
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)

    #Embedding (CNN or not):
    # Xtrain = tokenizer.texts_to_sequences(train_docs)
    # Xtest = tokenizer.texts_to_sequences(test_docs)
    #CNN:
    vocab_size = len(tokenizer.word_index) + 1
    Xtrain = pad_sequences(Xtrain, maxlen = input_length, padding='post')
    Xtest = pad_sequences(Xtest, maxlen = input_length, padding='post')

    #Embedding (CNN or not):
    # embeddings_index = dict()
    # f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
    # # f = open('GoogleNews-vectors-negative300.bin', mode='rb')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()
    # print('Loaded %s word vectors.' % len(embeddings_index))
    # # create a weight matrix for words in training docs
    # embedding_matrix = np.zeros((vocab_size, 100))
    # for word, i in tokenizer.word_index.items():
    #   embedding_vector = embeddings_index.get(word)
    #   if embedding_vector is not None:
    #     embedding_matrix[i] = embedding_vector


    #Embedding (CNN or not): embedding_matrix, vocab_size in outputs
    # AND  np.asarray for x train+test

    #CNN or Embedding: vocab_size in outputs
    return Xtrain, Xtest, tokenizer, vocab_size


def c_kappa(tps,fps,fns,tns):
    x = ((tps*tns) - (fns*fps))
    x=2*x
    y = 1 / (((tps+fps)*(fps+tns))+((tps+fns)*(fns+tns)))
    result = x*y
    return result


def bows_nlp(fold_num):
    #This is where I can adjust which column to train+test
    directory = STATS
    #This is where I can change the Vocab list
    acc, tps, fps, fns, tns, freqs = bows_stats(BIGRAM_VOCAB, fold_num, directory)
    acc=statistics.mean(acc)
    tps=statistics.mean(tps)
    fps=statistics.mean(fps)
    fns=statistics.mean(fns)
    tns=statistics.mean(tns)

    kap=c_kappa(tps,fps,fns,tns)
    print(kap)

    data = {'Data':directory, 'Fold':fold_num, 'Acc':acc, 'TP':tps,'FP':fps, 'FN':fns, 'TN':tns,'Freq':[freqs], 'Kappa':kap}
    #Typed in notes manually about NLP specifics, this is just for results

    master = 'Results_NLP.xlsx'
    data = pd.DataFrame.from_dict(data)
    mf=pd.read_excel(master)
    new=pd.concat([mf,data])
    newexcel=pd.DataFrame.to_excel(new, 'Results_NLP.xlsx', index=False)



def embed_stats(vocab,fold_num,directory):
    scores = list()
    tps = list()
    fps = list()
    fns = list()
    tns = list()
    freq = list()
    n_training_repeats = 1
    for i in range(n_training_repeats):
        train_docs, ytrain = load_clean_dataset(vocab, True, fold_num, directory)
        ytrain=np.asarray(ytrain)
        print(ytrain)
        input_length = max([len(v.split()) for v in train_docs])
        test_docs, ytest = load_clean_dataset(vocab, False, fold_num, directory)

        #Embedding (CNN or not): matrix, vocab_size after input_length in prepare_data()
        Xtrain, Xtest, tokenizer, vocab_size = prepare_data(train_docs, test_docs, 'binary', input_length)

        n_repeats = 10
        print(Xtest.shape)
        n_words = Xtest.shape[0]
        freq.append(sum(ytrain))
        for i in range(n_repeats):
            # define network

            #Embedding (CNN or not): matrix after input_length in define_model
            model = define_model(n_words, vocab_size, input_length)
            # fit network
            model.fit(Xtrain, ytrain, epochs=15, verbose=0)
            # evaluate
            dat, acc, tp, fp, fn, tn = model.evaluate(Xtest, ytest, verbose=0)
            print(dat)
            scores.append(acc)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
            print('%d accuracy: %s' % ((i+1), acc))
    return scores, tps, fps, fns, tns, (freq,len(ytrain))


def embed_nlp(fold_num):
    #change directory for other columns (global variables at the top)
    directory = EXPECTED
    print(directory)
    #Change vocab here for CNN+vocab
    acc, tps, fps, fns, tns, freqs = embed_stats(BIGRAM_VOCAB,fold_num, directory)
    acc=statistics.mean(acc)
    tps=statistics.mean(tps)
    fps=statistics.mean(fps)
    fns=statistics.mean(fns)
    tns=statistics.mean(tns)

    kap=c_kappa(tps,fps,fns,tns)
    print(kap)
    data = {'Data':directory, 'Fold':fold_num, 'Acc':acc, 'TP':tps,'FP':fps, 'FN':fns, 'TN':tns,'Freq':[freqs], 'Kappa':kap}
    master = 'Results_NLP.xlsx'
    data = pd.DataFrame.from_dict(data)
    mf=pd.read_excel(master)
    new=pd.concat([mf,data])
    newexcel=pd.DataFrame.to_excel(new, 'Results_NLP.xlsx', index=False)




def auto(one=True,two=True,three=True,four=True,five=True):
    # Sometimes freezes in the middle
    # 1-5 are folds held out for testing
    # bows_nlp is used only for bag of words without a CNN,
    # everything else is using embed_nlp

    if one:
        # bows_nlp(1)
        embed_nlp(1)
        print('First fold done')
    if two:
        # bows_nlp(2)
        embed_nlp(2)
        print('Second fold done')
    if three:
        # bows_nlp(3)
        embed_nlp(3)
        print('Third fold done')
    if four:
        # bows_nlp(4)
        embed_nlp(4)
        print('Fourth fold done')
    if five:
        # bows_nlp(5)
        embed_nlp(5)
        print('Fifth fold done')
    print('COMPLETE')
