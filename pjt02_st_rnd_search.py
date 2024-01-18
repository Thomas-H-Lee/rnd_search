################################################## 
#   PJT : Similarity-based Search and recommendation          #
#   Desc. :             #
#   Date : 2024-01-02                            #
#   Writer : Tom Lee                          #
#   Version : V0.1                               #
##################################################
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc
import base64
import streamlit.components.v1 as components
from elasticsearch import Elasticsearch

logo = Image.open(r'C:\Git\GitHub\rnd_search\mathworks_ci.png')

# Elasticsearch 셋업 : Consulting 서버 (App Search APIs - python용 Guide 준용)
_ES_URL = "http://203.245.157.103:9200"  # Linux Server 
_ES_INDEX = "pjt01_test_analyzer"
_DOC_TYPE = _ES_INDEX
es_client = Elasticsearch([_ES_URL])

## ## DEFINE CUSTOM ANALYZER & USER MAPPING
#PUT eng_custom_analyzer
# load preset json mapping file(_ES_setting.json)


engine_name = "poc-research-doc"

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


def disp_result(resp, idx_start):
    # 추천 결과 1
    for i in range(idx_start, idx_start+3):
        st.info(' 연관스코어 : {}'.format(resp['results'][i]['_meta']['score']))
        st.write('##### 기술명 : {}'.format(resp['results'][i]['title']['raw']))
        st.write('- 연구기간 : {} \n - 추진책임자 : {} \n - 활용부서 : {}\n- 활용책임자 : {}'.format(\
            resp['results'][i]['date']['raw'], \
            resp['results'][i]['leader']['raw'], \
            resp['results'][i]['user_dept']['raw'], \
            resp['results'][i]['user']['raw']))
        try:
            st.markdown('  - 기술설명 : {}'.format(resp['results'][i]['abstract']['snippet']))
        except:
            st.write('- 기술설명이 없습니다.')
        st.write('- 적용 현장 : \n - 적용 건수 : ')
        st.write('- 주제 키워드 : {} / {}'.format(resp['results'][i]['topic1']['raw'], \
                                                resp['results'][i]['topic2']['raw']))
        st.markdown("* * *")

def main():
    col1, col2 = st.columns( [0.9, 0.1])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-arial: 'Cooper Black'; color: #02559e;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">R&D센터 기술정보 검색모델</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )

    st.write("본 웹 서비스는 R&D센터 기술정보 검색모델로, 보유 기술 검색, 특허 검색을 위한 앱서비스 입니다.\n\n R&D센터 R&D기획그룹")    
    # st.write(templates.load_css(), unsafe_allow_html=True)

    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        search_word = st.text_input('')


    with col2:               # To display brand logo        
              
        st.write(' \n')
        st.write(' \n')
        # Run Button 
        clicked = st.button("Search")
    if clicked :
        # App Search 클라이언트 사용한 쿼리
        resp = app_search.search(
        engine_name = engine_name,
        body={
            "query": search_word
        }
        )
        # 검색결과 표시
        # To display 기술명
        # st.write('기술명 : {}'.format(resp))
        col1, col2 = st.columns( [0.3, 0.7])
        with col1:               # To display 기술명
            st.write(' \n')


        with col2:               # To display 기술 요약        
                
            st.write(' \n')
        # 추천 결과 1
        st.info(' 연관스코어 : {}'.format(resp['results'][0]['_meta']['score']))
        title = resp['results'][0]['title']['raw']
        b_link = resp['results'][0]['brochure_link']['raw']
        st.markdown('##### [추천 기술 1] : [{title}]({link})'.format(title=title, link=b_link))
        st.write('- 연구기간 : {} \n - 추진책임자 : {} \n - 활용부서 : {}\n- 활용책임자 : {}'.format(\
            resp['results'][0]['date']['raw'], \
            resp['results'][0]['leader']['raw'], \
            resp['results'][0]['user_dept']['raw'], \
            resp['results'][0]['user']['raw']))
        try:
            st.markdown('  - 기술설명 :')
            st.components.v1.html(resp['results'][0]['abstract']['snippet'], width=None, height=None, scrolling=False)
        except:
            st.write('   기술설명이 없습니다.')
        st.write('- 적용 현장 : \n - 적용 건수 : ')
        st.write('- 주제 키워드 : {} / {}'.format(resp['results'][0]['topic1']['raw'], \
                                                resp['results'][0]['topic2']['raw']))
        st.markdown("* * *")

        if resp['meta']['page']['total_results'] > 1:
        # 추천 결과 2
            st.info(' 연관스코어 : {}'.format(resp['results'][1]['_meta']['score']))
            title = resp['results'][1]['title']['raw']
            b_link = resp['results'][1]['brochure_link']['raw']
            st.markdown('##### [추천 기술 2] : [{title}]({link})'.format(title=title, link=b_link))
            st.write('- 연구기간 : {} \n - 추진책임자 : {} \n - 활용부서 : {}\n- 활용책임자 : {}'.format(\
                resp['results'][1]['date']['raw'], \
                resp['results'][1]['leader']['raw'], \
                resp['results'][1]['user_dept']['raw'], \
                resp['results'][1]['user']['raw']))
            try:
                st.markdown('  - 기술설명 :')
                st.components.v1.html(resp['results'][1]['abstract']['snippet'], width=None, height=None, scrolling=False)
            except:
                st.write('   기술설명이 없습니다.')
            st.write('- 적용 현장 : \n - 적용 건수 : ')
            st.write('- 주제 키워드 : {} / {}'.format(resp['results'][1]['topic1']['raw'], \
                                                    resp['results'][1]['topic2']['raw']))
            st.markdown("* * *")
    
        if resp['meta']['page']['total_results'] > 2:
        # 추천 결과 3
            st.info(' 연관스코어 : {}'.format(resp['results'][2]['_meta']['score']))
            title = resp['results'][2]['title']['raw']
            b_link = resp['results'][2]['brochure_link']['raw']
            st.markdown('##### [추천 기술 3] : [{title}]({link})'.format(title=title, link=b_link))
            st.write('- 연구기간 : {} \n - 추진책임자 : {} \n - 활용부서 : {}\n- 활용책임자 : {}'.format(\
                resp['results'][2]['date']['raw'], \
                resp['results'][2]['leader']['raw'], \
                resp['results'][2]['user_dept']['raw'], \
                resp['results'][2]['user']['raw']))
            try:
                st.markdown('  - 기술설명 :')
                st.components.v1.html(resp['results'][2]['abstract']['snippet'], width=None, height=None, scrolling=False)
            except:
                st.write('   기술설명이 없습니다.')
            st.write('- 적용 현장 : \n - 적용 건수 : ')
            st.write('- 주제 키워드 : {} / {}'.format(resp['results'][2]['topic1']['raw'], \
                                                    resp['results'][2]['topic2']['raw']))

        # if resp['meta']['page']['total_results'] > 3:
        #     clicked = st.button("더보기")
        #     if clicked :
        #         st.write('- 적용 현장 : \n - 적용 건수 : ')
                # disp_result(resp, 3)






        # if len(resp['hits']['hits']) ==0:
        #     st.error("Not found")
        #     st.text('검색결과가 없습니다.')
        # else:
        #     st.success("Successful")
        #     st.text("총 {}개의 결과가 검색되었습니다.".format(result['hits']['total']['value']))
        #     # brief_result = result['docbody'][0:50]
        #     #print(result)
        #     with st.container():
        #         st.markdown("* * *")
        #         # index name
        #         st.text('인덱스 이름 = {}'.format(result['hits']['hits'][0]['_index']))
        #         # doc id
        #         st.text('문서 ID = {}'.format(result['hits']['hits'][0]['_id']))
        #         # score
        #         st.text('연관스코어 = {}'.format(result['hits']['hits'][0]['_score']))
        #         # pjt name
        #         st.text('PJT명 = {}'.format(result['hits']['hits'][0]['_source']['pjtname']))
        #         # doctitle
        #         st.text('문서명 = {}'.format(result['hits']['hits'][0]['_source']['doctitle']))
        #         # doc class
        #         st.text('문서 종류 = {}'.format(result['hits']['hits'][0]['_source']['docclass']))
        #         # _source
        #         result_content = result['hits']['hits'][0]['_source']['docbody']
        #         result_content_brief = re.findall(r'.{20}<em>.{100}', result_content, re.I)
        #         st.text('문서 개요 = {}'.format(result_content_brief))
        #         # hits
        #         #st.text(result['hits'])
        #         # 판다스 DataFrame() 함수로 딕셔너리를 데이터프레임으로 변환, 변수 df에 저장
        #         df = pd.DataFrame.from_records([result['hits']['hits'][0]['_source']['df_table']])
        #         df = df.transpose()
        #         st.dataframe(df)
        #         st.text("2. 관련 Design Factor")
        #         st.text('Wind Load Code = {}'.format(result['hits']['hits'][0]['_source']['df_table']['df_wlc_value']))
        #         st.text('Basic Wind Design = {}'.format(result['hits']['hits'][0]['_source']['df_table']['df_ws_value']))
        #         st.text('Wind factor = {}'.format(result['hits']['hits'][0]['_source']['df_table']['df_ec_value']))
        #     if len(result['hits']['hits']) >=2:
        #         with st.container():
        #             st.markdown("* * *")
        #             # index name
        #             st.text('인덱스 이름 = {}'.format(result['hits']['hits'][1]['_index']))
        #             # doc id
        #             st.text('문서 ID = {}'.format(result['hits']['hits'][1]['_id']))
        #             # score
        #             st.text('연관스코어 = {}%'.format(result['hits']['hits'][1]['_score'])*10)
        #             # pjt name
        #             st.text('PJT명 = {}'.format(result['hits']['hits'][1]['_source']['pjtname']))
        #             # doctitle
        #             st.text('문서명 = {}'.format(result['hits']['hits'][1]['_source']['doctitle']))
        #             # doc class
        #             st.text('문서 종류 = {}'.format(result['hits']['hits'][1]['_source']['docclass']))
        #             # _source
        #             st.text('문서 개요 = {}'.format(result['hits']['hits'][1]['_source']['docbody'][300:400]))
        #             # hits
        #             #st.text(result['hits'])
        #             st.text("2. 관련 Design Factor")
        #             df = pd.DataFrame.from_records([result['hits']['hits'][1]['_source']['df_table']])
        #             df=df.transpose()
        #             st.dataframe(df)
        #             st.text('Wind Load Code = {}'.format(result['hits']['hits'][1]['_source']['df_table']['df_wlc_value']))
        #             st.text('Basic Wind Design = {}'.format(result['hits']['hits'][1]['_source']['df_table']['df_ws_value']))
        #             st.text('Wind factor = {}'.format(result['hits']['hits'][1]['_source']['df_table']['df_ec_value']))
        #         # dataframe
        #         if len(result['hits']['hits']) >=3:
        #             with st.container():
        #                 st.markdown("* * *")
        #                 # index name
        #                 st.text('인덱스 이름 = {}'.format(result['hits']['hits'][2]['_index']))
        #                 # doc id
        #                 st.text('문서 ID = {}'.format(result['hits']['hits'][2]['_id']))
        #                 # score
        #                 st.text('연관스코어 = {}'.format(result['hits']['hits'][2]['_score']))
        #                 # pjt name
        #                 st.text('PJT명 = {}'.format(result['hits']['hits'][2]['_source']['pjtname']))
        #                 # doctitle
        #                 st.text('문서명 = {}'.format(result['hits']['hits'][2]['_source']['doctitle']))
        #                 # doc class
        #                 st.text('문서 종류 = {}'.format(result['hits']['hits'][2]['_source']['docclass']))
        #                 # _source
        #                 st.text('문서 개요 = {}'.format(result['hits']['hits'][2]['_source']['docbody'][300:400]))
        #                 # hits
        #                 #st.text(result['hits'])
        #                 st.text("2. 관련 Design Factor")
        #                 df = pd.DataFrame.from_records([result['hits']['hits'][2]['_source']['df_table']])
        #                 df=df.transpose()
        #                 st.dataframe(df)
        #                 st.text('Wind Load Code = {}'.format(result['hits']['hits'][2]['_source']['df_table']['df_wlc_value']))
        #                 st.text('Basic Wind Design = {}'.format(result['hits']['hits'][2]['_source']['df_table']['df_ws_value']))
        #                 st.text('Wind factor = {}'.format(result['hits']['hits'][2]['_source']['df_table']['df_ec_value']))
if __name__ == "__main__":
    main()
