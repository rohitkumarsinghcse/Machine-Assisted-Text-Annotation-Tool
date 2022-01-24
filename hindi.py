def preprocess(fname):
    dataset=[]
    with open(fname,encoding='UTF-8') as f:
        for line in f:
            if not line.startswith('<'):
                for word in line.split(' '):
                    pair=word.split('_')
                    if len(pair)>=2:
                        dataset.append((pair[0],pair[1]))
    return dataset

preprocess('corpus_hindi.txt')