################################################## 
#   PJT : R&D 기술정보검색모델 구축                #
#   Desc. : 텍스트 추출 & 인덱싱(Appsearch)        #
#           연구과제 완료보고서 링크 추가 인덱싱 모듈     #
#   Date : 2022-06-30                            #
#   Writer : Hodong Lee                          #
#   Version : V1.2                               #
##################################################

#%%
import pandas as pd
import regex as re

from elastic_enterprise_search import AppSearch
import urllib.parse

# App Search 클라이언트 셋업 및 인덱스
# Setup Client
app_search = AppSearch(
    "203.245.157.104:3002/api/as/v1",
    http_auth="private-fcgi6zgirnwru77pun4g9bu1",
    use_https=False
)       

#%% 데이터 입력
    
# (1) 엑셀에서 대상 연구과제 리스트 가져 가져오기
rs_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\메타데이터_v5_220628_활용PJT추가.xlsx'  

df_rs = pd.read_excel(rs_file, index_col=False)    
n_docs_rs = len(df_rs.index)

## 기존 인덱스의 필드데이터로 부터 가공 후 필드 업데이트
engine_name = "poc-research-doc"


#%% 브로셔와 요약문 링크 인덱싱
idx_start = 0
idx_end = n_docs_rs
for i in range(idx_start, idx_end):
    # 대상 ID / 폴더 / 파일명
    target_id = df_rs.loc[i]['id']
    b_file = df_rs.loc[i]['brochure_link']
    s_file = df_rs.loc[i]['summary_link']
    # 파일 전체 경로
    b_link_folder = 'RND기술정보/브로셔/' # NAS 폴더명
    s_link_folder = 'RND기술정보/요약문/' # NAS 폴더명

    main_link = 'http://203.245.157.75:8080/download/'

    b_link = b_link_folder + str(b_file)
    s_link = s_link_folder + str(s_file)
    
    # Document 업데이트
    target = {
            'id' : int(target_id), 
            'brochure_link': main_link + urllib.parse.quote(b_link), #브로셔
            'summary_link': main_link + urllib.parse.quote(s_link) # 요약문
            }
    # 엘라스틱 인덱싱 생성(index_document), 업데이트(put_document)
    app_search.put_documents(engine_name=engine_name, documents=target)
    print("전체 Doc개수 : {}, 현재 인덱싱 ID : {}, 남은 Doc 개수 : {}".format(n_docs_rs, target_id, n_docs_rs-i-1))




# %%
