################################################## 
#   PJT : R&D 기술정보검색모델 구축               #
#   Desc. : LDA Gensim 활용 토픽모델링                           #
#   Date : 2022-07-13                            #
#   Writer : Hodong Lee                          #
#   Version : V0.1                               #
##################################################
#%% 
from concurrent.futures import process
import nltk
from nltk.data import load
from nltk import sent_tokenize, word_tokenize
from konlpy.tag import Mecab
from tqdm import tqdm
import re
# import pickle
# import csv
import pandas as pd
# import numpy as np
import csv

from collections import Counter

from gensim.models.ldamodel import LdaModel
# from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
# from gensim.models.callbacks import PerplexityMetric
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# import pyLDAvis.gensim_models as gensimvis
# import pyLDAvis
# from gensim.models.coherencemodel import CoherenceModel
# import matplotlib.pyplot as plt
from elastic_enterprise_search import AppSearch

# App Search 클라이언트 셋업 및 인덱스
# Setup Client
app_search = AppSearch(
"203.245.157.104:3002/api/as/v1",
http_auth="private-fcgi6zgirnwru77pun4g9bu1",
use_https=False
)  

def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    nouns = [s for s, t in tagged if t in ['NNG', 'NNP', 'VA', 'XR'] and len(s) >1]
    return nouns

def tokenize(df):
    tokenizer = Mecab(dicpath=r"c:/mecab/mecab-ko-dic")
    processed_data = []
    for sent in tqdm(df['content']):
        sentence = clean_text(str(sent).replace("\n", "").strip())
        processed_data.append(get_nouns(tokenizer, sentence))
    return processed_data
def clean_text(text):
    text = text.replace(".", "").strip()
    text = text.replace("·", " ").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
    text = re.sub(pattern=pattern, repl='', string=text)
    return text

def save_processed_data(processed_data):
    """
    토큰 분리한 데이터를 csv로 저장
    :param processed_data:
    :return:
    """
    with open('./tokenized_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)

def read_doc_appsearch(doc_id):
    engine_name = "poc-research-doc"
    try:
        current_doc = app_search.get_documents(engine_name=engine_name, \
        document_ids=[doc_id]
        )
        if current_doc[0] ==None:
            content = 'None'
        else:
            content = current_doc[0]["content"]
    except:
        content = 'None'
    data = pd.DataFrame({'id':doc_id, 'content' : content}, index=[0])
    return data

#%% 문서 내 빈출단어 추출
def extract_keywords(id):
    # 엘라스틱에서 해당 id로 문서 조회
    data = read_doc_appsearch(id)
    # 형태소 분리 & 명사만 선택
    processed_data = tokenize(data)
    # 각 단어별 출몰 횟수 카운트
    keyword_count = Counter(processed_data[0]) # 0 인덱스에 저장됨
    # 카운트 object를 데이터프레임으로 변환
    df_keyword = pd.DataFrame.from_dict(keyword_count, orient='index').reset_index()
    df_keyword.columns = ['word', 'count'] # 열명칭 변경
    df_keyword=df_keyword.sort_values('count', ascending=False).reset_index() # count 열에 따라 내림차순정렬
    df_keyword = df_keyword[0:10] # 10개 추출
    df_keyword = df_keyword.loc[:, 'word'] # word 컬럼만 추출
    df_keyword.values.tolist() # 리스트 변환
    return df_keyword


#%% 방법1 : gensim 활용
def lda_gensim(doc_id):
    
    

    # 단어사전 생성
    dictionary = corpora.Dictionary(processed_data)
    # 빈도가 5 이상인 단어와 전체의 50%로 이상 차지하는 단어는 필터링
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # 사전 속의 단어가 문장에서 몇 번 출현하는지 빈도 산정, 벡터화 (BOW) = "Corpus"
    corpus = [dictionary.doc2bow(text) for text in processed_data]


    num_topics = 5  # 생성될 토픽의 개수
    chunksize = 2000 # 한번의 트레이닝에 처리될 문서의 개수
    passes = 20 # 전체 코퍼스 트레이닝 횟수
    iterations = 400 # 문서 당 반복 횟수
    eval_every = None # 몇번의 pass마다 문서 Convergence평가할지
    


    temp = dictionary[0]
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    top_topics = model.top_topics(corpus) #, num_words=20)
    
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    
    ## 토픽 모델 검증
    # 방법1; 적정 토픽수를 정하기 위한 perplexity 검증  
    # perpelxity는 사전적으로는 혼란도 
    # 특정 확률 모델이 실제도 관측되는 값을 어마나 잘 예측하는지
    # Perlexity값이 작으면 토픽모델이 문서를 잘 반영
    # 작아지는것이 중요
    """
    perplexity_values=[]
    coherence_values=[]
    for i in range(2,15):
        ldamodel = LdaModel(corpus=corpus, num_topics =i, id2word=id2word) 

        # 방법1; 적정 토픽수를 정하기 위한 perplexity 검증  
        perplexity_values.append(ldamodel.log_perplexity(corpus))
        
        # 방법2; 적정 토픽수를 정하기 위한 Coherence 검증  
        coherence_mode_lda = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=id2word, topn=10, coherence='u_mass')
        coherence_lda = coherence_mode_lda.get_coherence()
        coherence_values.append(coherence_lda)
        
    # 차트 데이터 생성
    chart_data_per = pd.DataFrame(perplexity_values, columns=['perplexity'], index=None)
    chart_data_coh = pd.DataFrame(coherence_values, columns=['coherence'], index=None)
    print("perplexity_values= {}".format(perplexity_values))
    print("coherence_values= {}".format(coherence_values))
    # 차트 생성
    x = range(2,15)

    from bokeh.plotting import figure


    p = figure(
        title='coherence_values',
        x_axis_label='x',
        y_axis_label='y')

    p.line(x, coherence_values, legend_label='Trend', line_width=2)

    st.bokeh_chart(p, use_container_width=True)
    """
    # 차트 데이터 생성
    # 차트 생성
    
    # 방법2; 적정 토픽수를 정하기 위한 Coherence 검증  
    # coherence는 주제의 일관성을 측정
    # 해당 토픽모델이, 모델링이 잘 되었을수록 한 주제 안에는 의미론적으로 유사한 단어가 많이 모임
    # 상위 단어 간의 유사도를 계산하면 실제로 해당 주제가 의미론적으로 일치하는 단어들끼리 모여있는지 알 수 있음
  
    # from pprint import pprint
    # pprint(top_topics)
    # # html 형태 시각화    
    # lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    # pyLDAvis.save_html(lda_visualization, 'gensim_LDA.html')

    # return model, corpus, top_topics
#%%
if __name__ == '__main__':
    
    #%% 엑셀 파일에서 각 문서 id 읽어오기
    rs_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\메타데이터_v5_220630.xlsx'  

    df_rs = pd.read_excel(rs_file, index_col=False)    

    id = df_rs['id']
    doc_ids = id.values.tolist() #리스트변환

    # 모든 연구과제 보고서에 대해 실행
    
    for id in doc_ids:
        # 엘라스틱에서 해당 id로 문서 조회
        data = content_read(id)
        # 형태소 분리 & 명사만 선택
        processed_data = tokenize(data)


    lda_gensim(id)
# %%
