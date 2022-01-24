import nltk
nltk.download('treebank') 
nltk.download('universal_tagset')
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

with open("corpus.txt","w") as f:
    for data in nltk_data:
        line=""
        for d in data:
            line+=d[0]+"*"+d[1]+" "
        f.write(line+"\n")