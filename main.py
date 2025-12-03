# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit 앱 설정
st.set_page_config(
    page_title="운동 데이터 분석 웹페이지",
    layout="wide",
    initial_sidebar_state="expanded"
)

## 📌 함수 정의

@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 데이터를 클리닝하는 함수
    """
    # 데이터 로드
    df = pd.read_csv(file_path)

    # 필요한 전처리 및 클리닝
    # 1. '체지방율' 열이 NaN인 행 제거 (분석의 기준이므로)
    df.dropna(subset=['체지방율'], inplace=True)
    
    # 2. 분석에 필요한 숫자형 데이터만 추출
    # - 한글 이름이므로 `numeric_only=True`로 자동 선택이 어려워 주요 변수만 수동 선택하거나,
    # - `select_dtypes(include=np.number)`를 사용해 숫자형 열만 선택
    
    # 숫자형 데이터가 아닌 열 (문자열, 날짜, 코드 등) 제거
    numeric_df = df.select_dtypes(include=np.number)
    
    # 불필요하거나 고유성이 낮은 키/코드 열 제거 (예: 측정회차, 성별구분코드 등)
    cols_to_drop = [col for col in numeric_df.columns if len(df[col].unique()) < 10 and not col in ['나이', '신장', '체중']]
    numeric_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    # 체지방율이 0이거나 너무 극단적인 값인 행 제거 (이상치 처리)
    numeric_df = numeric_df[(numeric_df['체지방율'] > 0) & (numeric_df['체지방율'] < 50)]

    return numeric_df

## 🚀 메인 앱 로직

st.title("🏃‍♀️ 운동 데이터 상관관계 분석 웹페이지")
st.markdown("---")

# 1. 데이터 로드 및 전처리
file_name = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"
try:
    data = load_data(file_name)
    st.success(f"✅ 데이터 로드 성공: `{file_name}` (총 {len(data)}개 행)")

    # 2. 상관관계 분석
    st.header("📊 상관관계 분석")

    # 모든 변수 간의 상관관계 계산
    corr_matrix = data.corr()

    # '체지방율'과의 상관관계 추출 및 절대값 기준 정렬
    if '체지방율' in corr_matrix.columns:
        fat_corr = corr_matrix['체지방율'].sort_values(ascending=False)
        fat_corr_abs = fat_corr.abs().sort_values(ascending=False)
        
        # 체지방율 자기 자신 제외
        fat_corr_abs = fat_corr_abs.drop('체지방율')
        
        if not fat_corr_abs.empty:
            highest_corr_feature = fat_corr_abs.index[0]
            highest_corr_value = fat_corr[highest_corr_feature]
            
            st.info(f"💡 **체지방율**과 상관관계가 **가장 높은 속성**은 **`{highest_corr_feature}`**이며, 상관계수는 **`{highest_corr_value:.3f}`**입니다.")
            
            st.markdown("### 체지방율과의 상관관계 순위")
            st.dataframe(pd.DataFrame(fat_corr_abs).rename(columns={'체지방율': '상관계수 (절대값)'}).head(10))

            st.markdown("---")
            
            # 3. 산점도 (Scatter Plot) - 가장 높은 상관관계 속성
            st.subheader(f"📈 산점도: 체지방율 vs {highest_corr_feature}")
            
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=highest_corr_feature, y='체지방율', data=data, ax=ax_scatter)
            ax_scatter.set_title(f"체지방율과 {highest_corr_feature}의 관계", fontsize=16)
            ax_scatter.set_xlabel(highest_corr_feature, fontsize=12)
            ax_scatter.set_ylabel("체지방율", fontsize=12)
            st.pyplot(fig_scatter)
            
            st.markdown("---")
            
            # 4. 히트맵 (Heatmap) - 전체 변수
            st.subheader("🔥 히트맵: 전체 변수 간의 상관관계")
            
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, 
                        annot=False,  # 숫자는 너무 많아 생략
                        cmap='coolwarm', 
                        fmt=".2f",
                        linewidths=.5,
                        cbar_kws={'label': '상관계수'},
                        ax=ax_heatmap)
            ax_heatmap.set_title("전체 운동 데이터 속성 간의 상관관계", fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig_heatmap)
             # 히트맵 이미지 요청
            
        else:
            st.error("데이터에 분석할 수 있는 숫자형 속성이 충분하지 않습니다.")
    else:
        st.error("데이터에 '체지방율' 컬럼이 존재하지 않아 분석을 진행할 수 없습니다.")
        st.dataframe(data.head()) # 데이터 프레임의 상단 5행 표시하여 컬럼명 확인 유도
        
except FileNotFoundError:
    st.error(f"❌ 오류: 지정된 파일 `{file_name}`을 찾을 수 없습니다. 파일이 `app.py`와 같은 위치에 있는지 확인해 주세요.")
except Exception as e:
    st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 운동 데이터 분석 웹페이지")
