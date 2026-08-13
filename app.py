import streamlit as st
import pandas as pd
import numpy as np
import itertools
import joblib

# 画面設定
st.set_page_config(page_title="競艇 3連単確率予想シミュレーター", page_icon="🚤", layout="wide")
st.title("🚤 競艇 6艇データ入力 & 3連単確率予想シミュレーター")
st.write("6艇それぞれの「全国勝率」「平均ST」を入力すると、コースの有利不利（枠番）を加味して全120通りの3連単出現確率を自動試算します。")

# 保存したモデルの読み込み
@st.cache_resource
def load_models():
    models = joblib.load('boat_race_models.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return models, feature_columns

try:
    models, feature_columns = load_models()
except Exception as e:
    st.error("モデルファイルの読み込みに失敗しました。`boat_race_models.pkl` と `feature_columns.pkl` が同じフォルダにあるか確認してください。")
    st.stop()

# --------------------------------------------------
# 1. 6艇のデータ入力フォーム
# --------------------------------------------------
st.header("📋 出走表データの入力")

default_win_rates = [7.26, 6.17, 5.80, 6.10, 4.80, 4.50]
default_sts = [0.14, 0.14, 0.15, 0.14, 0.17, 0.16]

boats_data = []

cols1 = st.columns(3)
cols2 = st.columns(3)
all_cols = cols1 + cols2

for i in range(6):
    frame_num = i + 1
    with all_cols[i]:
        st.subheader(f"【{frame_num}号艇】")
        wr = st.number_input(f"{frame_num}号艇 勝率", min_value=0.00, max_value=10.00, value=default_win_rates[i], step=0.01, key=f"wr_{frame_num}")
        st_val = st.number_input(f"{frame_num}号艇 平均ST", min_value=0.00, max_value=0.50, value=default_sts[i], step=0.01, key=f"st_{frame_num}")
        
        # 特徴量作成（1号艇〜6号艇のワンホット）
        input_dict = {col: [0] for col in feature_columns}
        
        frame_key = f"frame_{frame_num}"
        if frame_key in input_dict:
            input_dict[frame_key] = [1]
            
        if 'win_rate' in input_dict:
            input_dict['win_rate'] = [wr]
        if 'avg_st' in input_dict:
            input_dict['avg_st'] = [st_val]
                
        input_df = pd.DataFrame(input_dict)[feature_columns]
        boats_data.append({
            'frame': frame_num,
            'win_rate': wr,
            'input_df': input_df
        })

# --------------------------------------------------
# 2. 予測実行ボタンと計算ロジック
# --------------------------------------------------
if st.button("🚀 3連単の確率を予測する", type="primary", use_container_width=True):
    
    boat_scores = {}
    for boat in boats_data:
        f = boat['frame']
        df_in = boat['input_df']
        wr = boat['win_rate']
        
        # AIモデルから算出された1着確率
        p1 = models['1着率'].predict_proba(df_in)[0][1]
        
        boat_scores[f] = {
            'p1': p1,
            'win_rate': wr
        }

    # 2着・3着に残りやすいコース補正値
    place_weight_2nd = {1: 1.00, 2: 0.85, 3: 0.75, 4: 0.65, 5: 0.45, 6: 0.30}
    place_weight_3rd = {1: 1.00, 2: 0.90, 3: 0.80, 4: 0.70, 5: 0.55, 6: 0.40}

    combos = list(itertools.permutations(range(1, 7), 3))
    raw_results = []
    total_raw_score = 0.0
    
    for c in combos:
        first, second, third = c
        
        # 1着: AIが学習したコース有利込みの1着確率
        # 2着: 選手の地力（勝率） × コース2着補正
        # 3着: 選手の地力（勝率） × コース3着補正
        score = (boat_scores[first]['p1']) * \
                (boat_scores[second]['win_rate'] * place_weight_2nd[second]) * \
                (boat_scores[third]['win_rate'] * place_weight_3rd[third])
                
        total_raw_score += score
        raw_results.append({'出目': f"{first}-{second}-{third}", 'raw_score': score})
    
    # 全120通りの確率算出（合計100%になるよう正規化）
    results_df = pd.DataFrame(raw_results)
    results_df['確率'] = (results_df['raw_score'] / total_raw_score) * 100
    results_df = results_df.sort_values(by='確率', ascending=False).reset_index(drop=True)
    results_df['順位'] = results_df.index + 1

    # --------------------------------------------------
    # 3. 結果の表示
    # --------------------------------------------------
    st.markdown("---")
    st.header("📊 3連単 予測確率ランキング")
    
    st.subheader("🔥 期待度 TOP 5")
    top5_cols = st.columns(5)
    for idx in range(5):
        row = results_df.iloc[idx]
        with top5_cols[idx]:
            st.metric(
                label=f"第 {row['順位']} 位",
                value=f"{row['出目']}",
                delta=f"{row['確率']:.2f}%"
            )
            
    st.subheader("📋 全120通り 確率一覧（高順位順）")
    display_df = results_df[['順位', '出目', '確率']].copy()
    display_df['確率'] = display_df['確率'].map('{:.2f}%'.format)
    st.dataframe(display_df, use_container_width=True, height=400)
