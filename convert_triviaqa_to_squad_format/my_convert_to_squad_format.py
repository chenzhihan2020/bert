import utils.utils

import utils.dataset_utils

import os

from tqdm import tqdm

import random

import nltk
nltk.download('punkt')

import argparse

#from sklearn import feature_extraction  

from sklearn.feature_extraction.text import TfidfTransformer
  
from sklearn.feature_extraction.text import CountVectorizer

import math


def get_text(qad, domain):

    local_file = os.path.join(args.web_dir, qad['Filename']) if domain == 'SearchResults' else os.path.join(args.wikipedia_dir, qad['Filename'])

    return utils.utils.get_file_contents(local_file, encoding='utf-8')
##
def get_num_words(para):
    words=para.split(' ')
    return(len(words)+1)
##


def throw_irrelevant_paragraphs(text,answer_list,span_length):
    def find_answer(para,answer_list):
        doc=para.lower()
        for answer_string_in_doc in answer_list: 
            index = doc.find(answer_string_in_doc.lower())
            if(index!=-1):
                return True
        return False
    
    result=[]
    paras = text.split('\n')
    i=0
    while(i<len(paras)):        
        if find_answer(paras[i],answer_list):
            total_num=get_num_words(paras[i])
            result.append(paras[i])
            i+=1
            while(i<len(paras) and total_num<span_length):
                result.append(paras[i])
                total_num+=get_num_words(paras[i])
                i+=1                
        else: 
            i+=1
    return ' \n '.join(result).strip()

def filter_test_using_tfidf(text,query,span_length, filename):
    result=[query]
    paras = text.split('\n')
    i=0
    while(i<len(paras)):
        total_num=get_num_words(paras[i])
        tmp_result=paras[i]
        i+=1
        while(i<len(paras) and total_num<span_length):
            tmp_result+=paras[i]
            total_num+=get_num_words(paras[i])
            i+=1
        result.append(tmp_result)

    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    tfidf=transformer.fit_transform(vectorizer.fit_transform(result))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    #word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        
    # calculate the similarity
    def calculate_sim(query_weight, doc_weight):
        cal_query = 0.0
        cal_doc = 0.0
        cal_cos = 0.0
        for i in range(len(query_weight)):
            cal_query += (query_weight[i]*100) ** 2
            cal_doc += (doc_weight[i]*100) ** 2
            cal_cos += (query_weight[i]*100) * (doc_weight[i]*100)
        if(cal_doc==0 or cal_query==0):
            return 0
        return cal_cos / math.sqrt(cal_query * cal_doc)
    
    query_weight = weight[0];
    score = []
    for i in range(1, len(weight)):
        score.append((calculate_sim(query_weight, weight[i]), i))
    score=sorted(
            score,
            key=lambda x:x[0],
            reverse=True)
        
    #Join the first k paras
    new_result = []
    k = len(score) / 10 + 1
    
    for i in range(k):
        new_result.append(result[score[i][1]])
    return ' \n '.join(new_result).strip()
##

def select_relevant_portion(text):

    paras = text.split('\n')

    selected = []

    done = False

    for para in paras:

        sents = sent_tokenize.tokenize(para)

        for sent in sents:

            words = nltk.word_tokenize(sent)

            for word in words:

                selected.append(word)

                if len(selected) >= args.max_num_tokens:

                    done = True

                    break

            if done:

                break

        if done:

            break

        selected.append('\n')

    st = ' '.join(selected).strip()

    return st





def add_triple_data(datum, page, domain):

    qad = {'Source': domain}
    if args.is_test:
        key_list=['QuestionId', 'Question']
    else:
        key_list=['QuestionId', 'Question', 'Answer']
        
    for key in key_list:

        qad[key] = datum[key]

    for key in page:

        qad[key] = page[key]

    return qad





def get_qad_triples(data):

    qad_triples = []

    for datum in data['Data']:

        for key in ['EntityPages', 'SearchResults']:

            for page in datum.get(key, []):

                qad = add_triple_data(datum, page, key)

                qad_triples.append(qad)
#               print (page["Filename"])
#           time.sleep(10)
    return qad_triples





def convert_to_squad_format(qa_json_file, squad_file):

    qa_json = utils.dataset_utils.read_triviaqa_data(qa_json_file)

    qad_triples = get_qad_triples(qa_json)



    random.seed(args.seed)

    random.shuffle(qad_triples)



    data = []
    k=0
    for i,qad in enumerate(tqdm(qad_triples)):

        qid = qad['QuestionId']

        text = get_text(qad, qad['Source'])
        
        question = qad['Question']

##      selected_text = select_relevant_portion(text)
        if(not args.is_test):
            selected_text=throw_irrelevant_paragraphs(text,qad['Answer']['Aliases'],args.max_num_tokens)
        else:
            selected_text=filter_test_using_tfidf(text, question, args.max_num_tokens,qad['Filename'])
            #continue
##

        para = {'context': selected_text, 'qas': [{'question': question, 'answers': []}]}
      

        qa = para['qas'][0]

        qa['id'] = utils.dataset_utils.get_question_doc_string(qid, qad['Filename'])

        qa['qid'] = qid


        if (not args.is_test):
            #ans_string, index = utils.dataset_utils.answer_index_in_document(qad['Answer'], text)
##
            indexs = utils.dataset_utils.answer_index_in_document(qad['Answer'], selected_text)
##
           #if index == -1:
##
            if (len(indexs)==0):
                print(qad['Filename'])
                k+=1
                continue
                #if qa_json['Split'] == 'train':
     
                 #   continue
##
            else:
                # qa['answers'].append({'text': ans_string, 'answer_start': index})
##
                qa['answers']=qad['Answer']['Aliases']

##
        data.append({'paragraphs': [para]})

        if qa_json['Split'] == 'train' and len(data) >= args.sample_size and qa_json['Domain'] == 'Web':

            break


    print('# invalid :')
    print (k)

    squad = {'data': data, 'version': qa_json['Version']}

    utils.utils.write_json_to_file(squad, squad_file)

    print ('Added', len(data))





def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--is_test',default=False, help='converting test file')

    parser.add_argument('--triviaqa_file', help='Triviaqa file')

    parser.add_argument('--squad_file', help='Squad file')

    parser.add_argument('--wikipedia_dir', help='Wikipedia doc dir')

    parser.add_argument('--web_dir', help='Web doc dir')

    parser.add_argument('--seed', default=10, type=int, help='Random seed')

    parser.add_argument('--max_num_tokens', default=800, type=int, help='Maximum number of tokens from a document')

    parser.add_argument('--sample_size', default=80000, type=int, help='Random seed')

    parser.add_argument('--tokenizer', default='tokenizers/punkt/english.pickle', help='Sentence tokenizer')

    args = parser.parse_args()

    return args





if __name__ == '__main__':

    args = get_args()

    sent_tokenize = nltk.data.load(args.tokenizer)

    convert_to_squad_format(args.triviaqa_file, args.squad_file)