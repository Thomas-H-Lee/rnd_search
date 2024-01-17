################################################## 
#   PJT : R&D 기술정보검색모델 구축                #
#   Desc. : 텍스트 추출 & 인덱싱(Appsearch)        #
#           0 : 연구과제 완료보고서 인덱스 모듈     #
#           1 : 특허 테이블 인덱스 모듈            #
#           2 : 불필요 인덱스 삭제 모듈      
#           3 : 기존 인덱스의 필드데이터로 부터 가공 후 필드 업데이트      #
#   Date : 2022-05-13                            #
#   Writer : Hodong Lee                          #
#   Version : V1.0                               #
##################################################
#%%
## 모듈 선택
# _CHOICE_ = 3  # 0:연구보고서, 1:특허,  2 : Delete Document,  3: idnex 내부에서 토픽 필드 추가

from encodings import utf_8
from pdfminer.high_level import extract_text
import nltk

import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)  # for xlrd option
xlrd.xlsx.Element_has_iter = True

import pandas as pd
import regex as re
import json

import textract
import docx2txt
import tika
tika.initVM()
from tika import parser

try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import zipfile

from elastic_enterprise_search import AppSearch
import DirInfor
# import pjt02_lda_topic 
import pjt02_lda_gensim
# App Search 클라이언트 셋업 및 인덱스
# Setup Client
app_search = AppSearch(
    "203.245.157.104:3002/api/as/v1",
    http_auth="private-fcgi6zgirnwru77pun4g9bu1",
    use_https=False
)       

## PPTX 추출 코드 
NAMESPACE = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
TEXT = NAMESPACE + 't'

#%%
# 연구완료보고서 Abstract추출용(2페이지(Abstract)만 추출)
def get_pptx_abs(file_path):
    pptx_doc = zipfile.ZipFile(file_path)
    text_list = []
    # 각 슬라이드 name들 추출
    nums = []
    for d in pptx_doc.namelist():
        if d.startswith("ppt/slides/slide"):
            nums.append(int(d[len("ppt/slides/slide"):-4]))
    s_format = "ppt/slides/slide%s.xml"
    slide_name_list = [s_format % x for x in sorted(nums)]
    # 슬라이드를 순회하며 텍스트 추출
    # 2 페이지만 추출[1]
    xml_content = pptx_doc.read(slide_name_list[1])
    tree = XML(xml_content)
        
    slide_text_list = []
    for node in tree.iter(TEXT):
        if node.text:
            slide_text_list.append(node.text)
    text_list.append("".join(slide_text_list))
    pptx_doc.close()

    txt_abs = '\n'.join(text_list)

    return txt_abs


def text_extract(file_path, file):
    # 확장자에 따라 텍스트 추출 모듈 선정            
    if file.endswith(".docx"):  ## Validation 완료 ##
        txt_extract = docx2txt.process(file_path)
    elif file.endswith(".doc"):   ## Validation 완료 ##
        txt_extract = textract.process(file_path, encoding='utf-8')
        txt_extract = txt_extract.decode('utf-8')  # converts from bytestring to string
    elif file.endswith(".pdf"):  ## Validation 완료 ##
        txt_extract = extract_text(file_path) 
    elif file.endswith(".pptx"):  ## Validation 완료 ##
        txt_extract = textract.process(file_path)  
        txt_extract = txt_extract.decode('utf-8')   
    elif file.endswith(".ppt"):  
        parsed = parser.from_file(file_path)
        #print(parsed["metadata"]) #To get the meta data of the file
        #print(parsed["content"]) # To get the content of the file    
        txt_extract = parsed["content"]     
    else:
        txt_extract ='None'
    
    # 추출 텍스트 사전확인
    print("Extracted text is ={} ".format(txt_extract[:50]))        
    
    # 전처리 : 줄바꿈 제거(시작/끝 모두)
    txt_extract = str(txt_extract).replace("\n", " ").replace("\r", "")
    
    return txt_extract
    
#==================================================================================================
#%%    
# (1) 엑셀에서 대상 연구과제 리스트 가져 가져오기
rs_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\메타데이터_v5_220630.xlsx'  
pat_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\특허(593건).xlsx' 

df_rs = pd.read_excel(rs_file, index_col=False)    
# with open(pat_file, mode="r", encoding='utf-8') as file:
df_pat = pd.read_excel(pat_file, index_col=False)

n_docs_rs = len(df_rs.index)
n_docs_pat = len(df_pat.index)

# 추출 리스트 전처리 
#   - 각 행을 딕셔너리로 해서 리스트 형태로 반환 
# list_extract = df_list.to_dict('records')
# n_docs = len(list_extract)
# print("전체 문서 갯수 : {}".format(n_docs)) 
  
 
## Create table for export
# d_factor_list = ['id','content']
# df_export = pd.DataFrame(columns = d_factor_list)
# 데이터 축적용 빈 데이터 프레임 생성
# tb_extract_tmp = pd.DataFrame(columns = d_factor_list)
  
#%% _CHOICE_ == 0:

# 선택 1 : 연구완료보고서 추출 색인
idx_start = 0
idx_end = 1   #n_docs_rs, # 색인하려는 번호

for i in range(idx_start, idx_end):
    
    # (2) id, file full path 추출
    id = df_rs.loc[i]['id']
    folder_path = df_rs.loc[i]['link']
    file_name = df_rs.loc[i]['file_name']
    file_path = folder_path + "/" + file_name
    if folder_path != "None":
        # (3) 텍스트 추출
        txt_extract = text_extract(file_path, file_name)
        # 0. 초기 불용어 처리
        # pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
        # txt_extract = re.sub(pattern=pattern, repl='', string=txt_extract)
        # (4) 토픽 분석
        if txt_extract != "None":
            tokens, feature_names, sorting = pjt02_lda_topic.lda_text(txt_extract)
            print("top topics : {}".format(tokens))
            tagged = nltk.pos_tag(tokens)
            content_origin = txt_extract
            # 크기 5의 토픽 저장용 빈 리스트
            str_result = list(0 for i in range(0,5))

            topics=range(5)
            feature_names=feature_names
            sorting=sorting
            topics_per_chunk=10
            n_words=5
            result = []
            for j in range(0, len(topics), topics_per_chunk):
                # for each chunk:
                these_topics = topics[j: j + topics_per_chunk]
                # maybe we have less than topics_per_chunk left
                len_this_chunk = len(these_topics)
                # print topic headers
                # print top n_words frequent words
                for j in range(n_words):
                    # try:
                    result_dic = {'Topic '+str(these_topics) : feature_names[sorting[these_topics, j]]}
                    result_df = pd.DataFrame(result_dic)
                    result.append(result_df)
                    # print(result)
                    result2 = feature_names[sorting[these_topics, j]]
                    print("these_topics : {}".format(result2))
                    str_join = " ".join(result2)
                    str_result[j] = str_join
                
            # 각 5개의 토픽 단어를 5개 그룹으로 저장
            topic1 = str_result[0]
            topic2 = str_result[1]
            topic3 = str_result[2]
            topic4 = str_result[3]
            topic5 = str_result[4]
        else: 
            content_origin = txt_extract
            topic1 = "N/A"
            topic2 = "N/A"
            topic3 = "N/A"
            topic4 = "N/A"
            topic5 = "N/A"
    else:
        content_origin = "N/A"
        topic1 = "N/A"
        topic2 = "N/A"
        topic3 = "N/A"
        topic4 = "N/A"
        topic5 = "N/A"

    # Reduce content size along to containable Payload limit : 10MB      
    print(len(content_origin))
    if len(content_origin) >= 50000:
        print('cropped')
        content = (content_origin[:30000*2] + '.....')  
    else:
        content = content_origin
        
    engine_name = "poc-research-doc"

    print("인덱싱 Document ID : {}".format(int(df_rs.loc[i]["id"])))
    
    target = {
            'id' : int(df_rs.loc[i]["id"]), #과제번호
            'tech_area_1' : df_rs.loc[i]["tech_area_lv1"], # 분야 대
            'tech_area_2' : df_rs.loc[i]["tech_area_lv2"], # 분야 중
            'tech_area_3' : df_rs.loc[i]["tech_area_lv3"], # 분야 소
            'proposal_no' : int(df_rs.loc[i]["proposal_no"]), #제안번호
            'title' : df_rs.loc[i]["title"], #과제명
            'date' : df_rs.loc[i]["date"], #연구기간
            'leader' : df_rs.loc[i]["leader"], #추진책임자
            'user_dept': df_rs.loc[i]["user_dept"], #활용부서
            'user' : df_rs.loc[i]["user"], #활용책임자
            'tech_grade' : df_rs.loc[i]["tech_grade"], #기술등급
            'trm' : df_rs.loc[i]["trm"],  #TRM상품
            'content': content, # 전문내용
            'topic1' : topic1, # 토픽1 
            'topic2' : topic2, # 토픽2 
            'topic3' : topic3, # 토픽3 
            'topic4' : topic4, # 토픽4 
            'topic5' : topic5, # 토픽5 """
            'brochure_link' : df_rs.loc[i]["brochure_link"]  #브로셔링크
            }
    """
    target = {
            'id' : int(df_rs.loc[i]["id"]), #과제번호
            'brochure_link' : df_rs.loc[i]["brochure_link"]  #브로셔링크
    }"""
    # 엘라스틱 인덱싱 업데이트(*신규 index : index_documents, 업뎃 index : put_documents)
    app_search.put_documents(engine_name=engine_name, documents=target)
    print("현재 인덱싱 ID : {}, 남은 파일 개수 : {}".format(target['id'], n_docs_rs-i-1))


#%% _CHOICE_ == 1:
        
## 특허
# 엔진 설정
engine_name = "poc-patent-item"

for i in range(0, n_docs_pat):
    # 데이터프레임에서 한 행씩 추출
    df_row = df_pat.iloc[i]
    # 한 행을 json 구조로 변환(str형식이 됨)
    json_row = df_row.to_json()
    # json변환으로 유니코드로 된 형태를 utf8로 변환
    json_pat = json.dumps(json_row, ensure_ascii=False).encode('utf8')
    # string 형태 유니코드에서 UTF–8 바이트 코드로 변환 
    target = json_row.encode()  
    
    app_search.index_documents(engine_name=engine_name, documents=target)


#%% _CHOICE_ == 2:
# Document 삭제
engine_name = "poc-research-doc"

app_search.delete_documents(engine_name=engine_name,
document_ids=['20200043',
'20200030',
'20190028',
'20190017',
'20190014',
'20180096',
'20180091',
'20180066',
'20180064',
'20180051',
'20180047',
'20180043',
'20180019',
'20180011',
'20180008',
'20180007',
'20180005',
'20170009',
'20140040',
'20140036',
'20140019',
'20130088',
'20110033'])

#%% _CHOICE_ ==3:    
## 기존 인덱스의 필드데이터로 부터 가공 후 필드 업데이트
engine_name = "poc-research-doc"
##
# 조회
current_doc = app_search.list_documents(engine_name=engine_name)
n_docs = current_doc["meta"]["page"]["total_results"]

# idx_start = current_doc["results"][0]['id']
# idx_end = current_doc["results"][n_docs-1]['id']

#n_docs = 3
for i in range(0, n_docs):
    # 대상 ID
    id = current_doc["results"][i]["id"]
    # content 필드
    txt_extract = current_doc["results"][i]["content"]
    
    # 1) 주요 키워드 추가(빈출명사 10개 추출)
    keywords = pjt02_lda_gensim.extract_keywords(id) # 리스트 형태 반환

    keywords = "|".join(keywords) # 스트링 형태로 합침

    if len(txt_extract) != 0:
        target = {
                'id' : id, #과제번호
                'topic1': keywords
                }



    # 2) Abstract 추가
    # id = current_doc["results"][i]["id"]
    # txt_abstract = re.findall(r'^Abstract(.*?)(?=Contents)', content, re.M + re.S)  
    #file_name = df_rs.loc['id']['file_name']
    # Document 업데이트
    # if len(txt_abstract) != 0:
    #     target = {
    #             'id' : id, #과제번호
    #             'abstract': txt_abstract
    #             }

                
                

        # 엘라스틱 인덱싱 생성(index_document), 업데이트(put_document)
        app_search.put_documents(engine_name=engine_name, documents=target)
        print("현재 인덱싱 ID : {}".format(id))
    else:
        print("Abstract 추출 텍스트: None, 문서 ID : {}".format(id))

# %% _CHOICE_ ==4:    
## 추가 필드 업데이트(applied_pjt, applied_num) 
#%%
engine_name = "poc-research-doc"
# 열 데이터 타입 변경
#df_rs.astype({'applied_num': 'str'})

for i in range(0, n_docs_rs):
    target = {
            'id' : int(df_rs.loc[i]["id"]), #과제번호
            'applied_pjt': df_rs.loc[i]['applied_pjt'], #적용 PJT
            'applied_num': int(df_rs.loc[i]['applied_num'])  #적용 건수
            
            }
                
    # 엘라스틱 인덱싱 생성(index_document), 업데이트(put_document)
    app_search.put_documents(engine_name=engine_name, documents=target)
    print("현재 인덱싱 ID : {}".format(id), '\r')


