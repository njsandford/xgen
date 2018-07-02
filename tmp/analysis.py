#!/usr/bin/env python3

from Document import Document
from Corpus import Corpus
from os import listdir
from os.path import isfile, join, splitext, split

#fql = freqList name
def tfidf(document, corpus, fql, term):
    return document.tf(fql,term) * corpus.idf(fql,term)

def process_twitter_folder(corpus, folder, metadata):
    textfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".txt")]
    # textfiles = textfiles[:2] #limit for quick processing, remove to include all.
    for tf in textfiles:
        textname = splitext(split(tf)[1])[0] #extract just username from filename.
        print('Processing ' + textname)
        document = Document(textname, metadata)
        document.process_document(tf)
        corpus.add_document(document)

if __name__ == '__main__':
    # corpus = Corpus("MPs")
    con_corpus = Corpus("conservative") # Create corpus for all conservative tweets
    lab_corpus = Corpus("Labour") # Create corpus for all labour tweets
    process_twitter_folder(con_corpus, "mps/conservative", {'party': 'conservative'})
    process_twitter_folder(lab_corpus, "mps/labour", {'party': 'labour'})

    for doc in con_corpus.documents:
        print()
        print(doc.metadata['party'] + ": " + doc.name)
        print("Bag of words")
        for key,val in doc.fqls['bow'].most_common(10): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
            print(str(key) + ":" + str(val))

    for doc in lab_corpus.documents:
        print()
        print(doc.metadata['party'] + ": " + doc.name)
        print("Bag of words")
        for key,val in doc.fqls['bow'].most_common(10): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
            print(str(key) + ":" + str(val))

    print()
    print("conservative Corpus Bag of words")
    for key,val in con_corpus.fql_totals['bow'].most_common(10): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
        print(str(key) + ":" + str(val))
    con_corpus.fql_totals['bow'].plot(50, cumulative=False)

    print()
    print("Labour Corpus Bag of words")
    for key,val in lab_corpus.fql_totals['bow'].most_common(10): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
        print(str(key) + ":" + str(val))
    lab_corpus.fql_totals['bow'].plot(50, cumulative=False)

    #run through conservative docs, calculate tfidfs, and print top 10.
    for doc in con_corpus.documents:
        print()
        print(doc.metadata['party'] + ": " + doc.name)
        tfidfs = {}
        for term in doc.fqls['bow']:
            tfidfs[term] = tfidf(doc, con_corpus, 'bow', term) #calculate tfidf for this term/word.

        sorted_tfids = [(k, tfidfs[k]) for k in sorted(tfidfs, key=tfidfs.get, reverse=True)] #sort reverse order by tfidf.

        print("tfidfs")
        for key,val in sorted_tfids[:10]: #print top 10.
            print(str(key) + ":" + str(val))

    #run through labour docs, calculate tfidfs, and print top 10.
    for doc in lab_corpus.documents:
        print()
        print(doc.metadata['party'] + ": " + doc.name)
        tfidfs = {}
        for term in doc.fqls['bow']:
            tfidfs[term] = tfidf(doc, lab_corpus, 'bow', term) #calculate tfidf for this term/word.

        sorted_tfids = [(k, tfidfs[k]) for k in sorted(tfidfs, key=tfidfs.get, reverse=True)] #sort reverse order by tfidf.

        print("tfidfs")
        for key,val in sorted_tfids[:10]: #print top 10.
            print(str(key) + ":" + str(val))


    print(con_corpus.feature_lists['avg_word_length']) #just print average word lengths.
    print(lab_corpus.feature_lists['avg_word_length']) #just print average word lengths.
