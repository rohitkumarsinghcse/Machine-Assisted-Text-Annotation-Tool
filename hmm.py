from os import stat
import random
import numpy as n
import pandas as pd
import time
import hindi

random.seed(88)

def preprocess(fname):
    dataset=[]
    with open(fname) as f:
        for line in f:
            line=line.split(' ')
            for pair in line:
                pair=pair.split('*')
                if len(pair)==2:
                    dataset.append((pair[0],pair[1]))
    return dataset

def split_dataset(data,test_set_ratio):
    random.shuffle(data)    
    test_count=int(len(data)*test_set_ratio)
    return data[test_count:],data[:test_count]

# compute Emission Probability
def word_given_tag(word, tag, train_set):
    tag_list = [pair for pair in train_set if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_set
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

def count_t2_given_t1(t2, t1, train_set):
    tags = [pair[1] for pair in train_set]
    #Number of times first tag occurs
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        #Number of times second tag occurs given first tag
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

def get_transition_matrix(tags,train_set):
    #matrix m of n*n,where n=no of tags
    # m stores the m(i,j)=P(jth tag after the ith tag)
    m = n.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)):
            count_t2_t1,count_t1=count_t2_given_t1(t2,t1,train_set)
            m[i, j] = count_t2_t1/count_t1
    return m


def Viterbi(words, train_set,tags_df,isEnglish):
    state = []
    T = tags_df.columns #list(set([pair[1] for pair in train_set]))
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                if isEnglish:
                    transition_p = tags_df.loc['.', tag]
                else:
                    transition_p = tags_df.loc['', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
            # compute emission and state probabilities
            wgt=word_given_tag(words[key],tag,train_set)
            emission_p = wgt[0]/wgt[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))

def random_test_select(test_set,count):
    # choose random numbers of length count
    rndom = [random.randint(1,len(test_set)) for x in range(count)]
    
    # list tagged words
    test_run_base = [test_set[i] for i in rndom]
    
    # list of untagged words
    test_tagged_words = [tup[0] for tup in test_run_base]

    return test_run_base,test_tagged_words


def test(train_set,test_set,tags_df,count,isEnglish):
    test_run_base,test_tagged_words=random_test_select(test_set,count)

    start = time.time()
    tagged_seq = Viterbi(test_tagged_words,train_set,tags_df,isEnglish)
    end = time.time()
    difference = end-start
    
    print('Time taken in seconds: ', difference)
    
    # accuracy
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
    
    accuracy = len(check)/len(tagged_seq)
    print('Viterbi Algorithm Accuracy: ',accuracy*100)

def call_test(tmatrix,train_set,test_set,tags_df,isEnglish):
    if tmatrix is not None:
        no=int(input('Test count:'))
        test(train_set,test_set,tags_df,no,isEnglish)

def test_for_sentence(train_set,tags_df,isEnglish):
    sentence=input('Enter a sentence:')
    words=sentence.split(' ')
    print('Number of words =',len(words))
    print(Viterbi(words, train_set,tags_df,isEnglish))

def main():
    tmatrix=None
    train_set=None
    test_set=None
    tags_df=None

    tmatrix_h=None
    train_set_h=None
    test_set_h=None
    tags_df_h=None

    while True:
        if tmatrix is None:
            print('!!Train First!!\n1. Train')
        else:
            print('1. Train')
            print('2. Test English')
            print('3. Generate POS for a English sentence')
            print('4. Test Hindi')
            print('5. Generate POS for a Hindi sentence')
        print('6. Exit')
        ip=input('>')

        if ip=='1':
            dataset=preprocess('corpus.txt')
            print("\t>processed english corpus")
            dataset_h=hindi.preprocess('corpus_hindi.txt')
            print("\t>processed hindi corpus")

            train_set_h,test_set_h=split_dataset(dataset_h,0.1)

            train_set,test_set=split_dataset(dataset,0.1)

            tags=set()
            for p in dataset:
                tags.add(p[1])
            
            tags_h=set()
            for p in dataset_h:
                tags_h.add(p[1])

            print("\t>training for English")
            tmatrix=get_transition_matrix(tags,train_set)
            print("\t>training for hindi")
            tmatrix_h=get_transition_matrix(tags_h,train_set_h)

            tags_df = pd.DataFrame(tmatrix, columns = list(tags), index=list(tags))
            tags_df_h = pd.DataFrame(tmatrix_h, columns = list(tags_h), index=list(tags_h))

        elif ip=='2':
            call_test(tmatrix,train_set,test_set,tags_df,True)
        
        elif ip=='3':
            test_for_sentence(train_set,tags_df,True)

        elif ip=='4':
            call_test(tmatrix_h,train_set_h,test_set_h,tags_df_h,False)

        elif ip=='5':
            test_for_sentence(train_set_h,tags_df_h,False)
        
        elif ip=='6':
            break

main()