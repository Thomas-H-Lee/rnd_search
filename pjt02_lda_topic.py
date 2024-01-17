################################################## 
#   PJT : R&D 기술정보검색모델 구축               #
#   Desc. : 토픽모델링                           #
#   Date : 2022-05-03                            #
#   Writer : Hodong Lee                          #
#   Version : V0.1                               #
##################################################
# 
import streamlit as st

import nltk
from nltk import sent_tokenize, word_tokenize
#nltk.download('punkt') # 최초 한번만 설치 필요
from konlpy.tag import Mecab
import docx2txt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction import text 
from sklearn.decomposition import LatentDirichletAllocation
import mglearn 
import glob
import re

from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# 방법1 : gensim 활용
def lda_gensim(textstr):
    text = textstr.replace(".", "").strip()
    text = text.replace("·", " ").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
    text = re.sub(pattern=pattern, repl='', string=text)
    
    tokens = word_tokenize(text)

    # 2. 품사태깅
    #nltk.download('averaged_perceptron_tagger')
    tagged = nltk.pos_tag(tokens)

    num_topics = 5  # 생성될 토픽의 개수
    chunksize = 2000 # 한번의 트레이닝에 처리될 문서의 개수
    passes = 20 # 전체 코퍼스 트레이닝 횟수
    iterations = 400 # 문서 당 반복 횟수
    eval_every = 1 # 몇번의 pass마다 문서 Convergence평가할지

    dictionary = corpora.Dictionary(tagged)
    # 빈도가 5 이상인 단어와 전체의 50%로 이상 차지하는 단어는 필터링
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # 사전 속의 단어가 문장에서 몇 번 출현하는지 빈도 산정, 벡터화 (BOW) = "Corpus"
    corpus = [dictionary.doc2bow(text) for text in tagged]

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
  
    from pprint import pprint
    pprint(top_topics)
    # html 형태 시각화    
    lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_visualization, 'gensim_LDA.html')

    return model, corpus, top_topics

# 방법2 : sklearn 활용
def lda_text(textstr):
    
    # 0. 초기 불용어 처리
    pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
    textstr = re.sub(pattern=pattern, repl='', string=textstr)

    # 1. 문장/단어 토큰화
    #mecab = Mecab(r'C:\Users\delta02\AppData\Local\Programs\Python\Python39\Lib\site-packages')
    #mecab.pos(textstr)
    sentences = sent_tokenize(textstr)
    tokens = word_tokenize(textstr)
    # print("1. words token =", tokens)
    # print("1. token length:", len(tokens))

    # 2. 벡터화
    vect = CountVectorizer()
    vect.fit(tokens)

    X_train = vect.transform(tokens)
    # print("벡터화된 tokens:", repr(X_train))

    # print("2. 어휘사전의 크기:", len(vect.vocabulary_))
    # print("2. 어휘사전의 내용:", vect.vocabulary_)

    bag_of_words = vect.transform(tokens)
    # print("2. BOW:", repr(bag_of_words))
    # print("2. BOW의 밀집표현:", bag_of_words.toarray())

    # 3. 불용어 처리
    # print("3. 불용어 개수:", len(ENGLISH_STOP_WORDS))
    # print("3. 불용어 샘플:\n", list(ENGLISH_STOP_WORDS)[::10])
    # min_df : removing terms that appear too infrequently
    # max_df : removing terms that appear too frequently, also known as "corpus-specific stop words".
    my_additional_stop_words =['01', '02','03','04','05','06','07','08','09','10','21']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    # 문서에서 토큰이 나타난 횟수(min_df, max_df)
    vect = CountVectorizer(min_df=5, stop_words=stop_words).fit(tokens)
    t_text = vect.transform(tokens)

    # print("3. 어휘사전의 크기:", len(vect.vocabulary_))
    # print("3. 어휘사전의 내용:", vect.vocabulary_)

    # 4. 토픽 모델링 LDA
    lda = LatentDirichletAllocation(n_components=10, learning_method="batch", max_iter=25, random_state=0)
    doc_topics = lda.fit_transform(t_text)

    # print("4. lda.components_.shape:", lda.components_.shape)
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    print("sorting={}".format(sorting))
    feature_names = np.array(vect.get_feature_names())
    print("feature_names={}".format(feature_names))
    # print("tokens={}".format(tokens))

    return tokens, feature_names, sorting


if __name__ == "__main__":
    
    # Corpus
    ## 대상 파일 리스트
    get_source_list = glob.glob('C:/project/teaV2/scd/poscoenc/backup_working/*')
    pjt_source_list = [file for file in get_source_list if file.endswith(".txt")]
    n_files = len(pjt_source_list)
    print("source list : ".format(pjt_source_list))
    print("전체 파일 개수 : {}".format(n_files))

    pjt_source_list = ['D:\plant_arch_service\Workspace\연구과제 완료보고서_PJT데이터 활용률 향상을 위한 PJT수행문서 맞춤 추천 시스템 구축(1-2).docx']
    # 1. Extract text
    textstr = docx2txt.process(pjt_source_list[0])

    # 2. LDA분석 실시
    tokens, feature_names, sorting = lda_text(textstr)

    # 3. 10개 토픽 선정
    mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)

    # %%
