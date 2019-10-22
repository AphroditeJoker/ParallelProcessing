import sys
import argparse
import numpy as np
from pyspark import SparkContext

def toLowerCase(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')
    input_file = sc.textFile(args.input)

    
    if args.mode=='TF' :
        words = input_file.flatMap(lambda line:line.split())
        words = words.map(lambda word: toLowerCase(word))
        words = words.map(lambda word: stripNonAlpha(word))\
                     .filter(lambda word: "" not in word)\
                     .map(lambda word: (word,1))\
                     .reduceByKey(lambda x,y: x+y)
        words.saveAsTextFile(args.output)

    if args.mode=='TOP':
        #Transform into tuples:
        words_mf = input_file.map(lambda input_file:eval(input_file))\
                             .sortBy(lambda x: x[1], ascending=False)\
                             .take(20)
        file_output = open(args.output, 'w+')
        for i in words_mf:
            file_output.write(str(i) + '\n')
        file_output.close()



 
    def myfunc(lin):
        s = set()
        for word in lin:
            word = toLowerCase(word)
            word = stripNonAlpha(word)
        if word != "":
            s.add(word)
        return s
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        ls_files = sc.wholeTextFiles(args.input)
        num_files = ls_files.count()
        words = ls_files.map(lambda line: (line[0],line[1].split()))\
                             .flatMap(lambda line: (myfunc(line[1])))\
                             .map(lambda word:(word,1))\
                             .reduceByKey(lambda x,y : x+y)\
                             .map(lambda line: (line[0],np.log(1.0*num_files/line[1])))
        words.saveAsTextFile(args.output)
    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        rdd_tf = sc.textFile(args.input)\
                   .map(lambda line:eval(line))
        rdd_idf = sc.textFile(args.idfvalues)\
                    .map(lambda line:eval(line))
        new_rdd = rdd_tf.join(rdd_idf)\
                        .map(lambda (x,y):(x,(y[0]*y[1])))\
                        .sortBy(lambda (x,y):y, ascending=False)
        new_rdd.saveAsTextFile(args.output)
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        input1_rddTF = sc.textFile(args.input)\
                        .flatMap(lambda x: x.split())\
                        .map(toLowerCase)\
                        .map(stripNonAlpha)\
                        .filter(lambda x: x != '')\
                        .map(lambda x: (x, 1.))\
                        .reduceByKey(lambda x,y:x+y)
        input1_rddIDF = sc.textFile("anc.idf").map(eval)
        input1IDF = input1_rddIDF.join(input1_rddTF).mapValues(lambda x: x[0]*x[1])
        input2_rddTF = sc.textFile(args.other)\
                        .flatMap(lambda x: x.split())\
                        .map(toLowerCase).map(stripNonAlpha) \
                        .filter(lambda x: x != '')\
                        .map(lambda x: (x, 1.))\
                        .reduceByKey(lambda x,y:x+y)
        input2_rddIDF = sc.textFile("anc.idf").map(eval)
        input2IDF = input2_rddIDF.join(input2_rddTF).mapValues(lambda x: x[0]*x[1])
        TFIDF = input2IDF.join(input2IDF).values()
        x_1 = TFIDF.map(lambda x: x[0]*x[1]).sum()
        Input1_value = input1IDF.values()\
                    .map(lambda x: x**2)\
                    .sum()
        Input2_value = input2IDF.values()\
                    .map(lambda x: x**2)\
                    .sum()
        x_2 = np.sqrt(Input1_value * Input2_value)
        print x_1/x_2
