################################################## 
#   PJT : R&D 기술정보검색모델 구축                #
#   Desc. : 텍스트 추출 & 인덱싱(Appsearch)        #
#           연구과제 완료보고서 서론 추출 모듈     #
#   Date : 2022-05-19                            #
#   Writer : Hodong Lee                          #
#   Version : V1.0                               #
##################################################

#%%
# from unittest.case import DIFF_OMITTED
import pandas as pd
import regex as re
import json

from elastic_enterprise_search import AppSearch
       
# App Search 클라이언트 셋업 및 인덱스
# Setup Client
app_search = AppSearch(
    "203.245.157.104:3002/api/as/v1",
    http_auth="private-fcgi6zgirnwru77pun4g9bu1",
    use_https=False
)       



#%%
## PPTX 추출 코드 
try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import zipfile

NAMESPACE = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
TEXT = NAMESPACE + 't'
## PPTX 추출 코드 
NAMESPACE = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
TEXT = NAMESPACE + 't'

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

#%%	


#==================================================================================================
    
# (1) 엑셀에서 대상 연구과제 리스트 가져 가져오기
rs_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\메타데이터_v3_220512.xlsx'  

df_rs = pd.read_excel(rs_file, index_col=False)    
n_docs_rs = len(df_rs.index)

## 기존 인덱스의 필드데이터로 부터 가공 후 필드 업데이트
engine_name = "poc-research-doc"
#%%
# 조회
current_doc = app_search.list_documents(engine_name=engine_name)
n_docs = current_doc["meta"]["page"]["total_results"]
#%%

folder_path = df_rs.loc[96]['link']
file_name = df_rs.loc[96]['file_name']
print(file_name)


#%%
print(n_docs)
def find_idx(id_t):
    id_t = '20200053'
    for i in range(0, n_docs):
        if current_doc["results"][i]["id"] == id_t:
            print('i={}, for id={}'.format(i, id_t))
            break
    return i


#%%
#n_docs = 3
idx_start = 0
idx_end = n_docs
for i in range(idx_start, idx_end):
    # 대상 ID
    target_id = current_doc["results"][i]["id"]
    # content 필드
    txt_extract = current_doc["results"][i]["content"]
    # 2) Abstract 추가
    id = current_doc["results"][i]["id"]
    # txt_abstract = re.findall(r'^Abstract(.*?)(?=Contents)', txt_extract, re.M + re.S)  
    # 확장자 확인 후 abstract 추출
    print(txt_extract[:3000])
    file_name = df_rs.loc[i]['file_name'] 

    if file_name.endswith(".pptx"):
        folder_path = df_rs.loc[i]['link']
        file_name = df_rs.loc[i]['file_name']
        file_path = folder_path + '/' + file_name
        txt_abstract = get_pptx_abs(file_path)
        # txt_abstract = re.findall(r'Abstract(.+?)Contents', txt_extract, re.M + re.S)  
    else:
        txt_abstract = re.findall(r'Abstract(.*?)Contents', txt_extract, re.M + re.S)  

    print(txt_abstract)

 
    # Document 업데이트
    if len(txt_abstract) != 0:
        target = {
                'id' : id, #과제번호
                'abstract': txt_abstract
                }
        # 엘라스틱 인덱싱 생성(index_document), 업데이트(put_document)
        app_search.put_documents(engine_name=engine_name, documents=target)
        print("현재 인덱싱 ID : {}".format(id))
    else:
        print("Abstract 추출 텍스트: None, 문서 ID : {}".format(id))

# %%
