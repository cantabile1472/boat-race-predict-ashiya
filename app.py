import streamlit as st
import pandas as pd
import numpy as np
import itertools
import joblib

# 画面設定
st.set_page_config(page_title="競艇 3連単確率予想シミュレーター", page_icon="🚤", layout="wide")
st.title("🚤 競艇 6艇データ入力 & 3連単確率予想シミュレーター")
st.write("6艇それぞれの「全国勝率」「平均ST」を入力すると、全120通りの3連単出現確率を自動試算します。")

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

# デフォルト値の設定（テスト用数値）
default_win_rates = [6.80, 5.50, 5.20, 6.10, 4.80, 4.50]
default_sts = [0.13, 0.15, 0.16, 0.14, 0.17, 0.16]

boats_data = []

# 3列ずつ2行で入力欄を整理
cols1 = st.columns(3)
cols2 = st.columns(3)
all_cols = cols1 + cols2

for i in range(6):
    frame_num = i + 1
    with all_cols[i]:
        st.subheader(f"【{frame_num}号艇】")
        wr = st.number_input(f"{frame_num}号艇 勝率", min_value=0.00, max_value=10.00, value=default_win_rates[i], step=0.01, key=f"wr_{frame_num}")
        st_val = st.number_input(f"{frame_num}号艇 平均ST", min_value=0.00, max_value=0.50, value=default_sts[i], step=0.01, key=f"st_{frame_num}")
        
        # 特徴量データの作成（モデル学習時の列名に完全一致させる）
        input_dict = {col: [0] for col in feature_columns}
        
        # 1号艇〜6号艇 または frame_1〜frame_6 の両方のワンホット命名に対応
        if f"{frame_num}号艇" in input_dict:
            input_dict[f"{frame_num}号艇"] = [1]
        elif f"frame_{frame_num}" in input_dict:
            input_dict[f"frame_{frame_num}"] = [1]
            
        # 勝率・平均STの列名（日本語・英語どちらでも吸収できるように設定）
        if '全国勝率' in input_dict:
            input_dict['全国勝率'] = [wr]
        elif 'win_rate' in input_dict:
            input_dict['win_rate'] = [wr]
            
        if '平均ST' in input_dict:
            input_dict['平均ST'] = [st_val]
        elif 'avg_st' in input_dict:
            input_dict['avg_st'] = [st_val]
        
        boats_data.append({
            'frame': frame_num,
            'input_df': pd.DataFrame(input_dict)[feature_columns] # 列の並び順も正しく揃える
        })

# --------------------------------------------------
# 2. 予測実行ボタンと計算ロジック
# --------------------------------------------------
if st.button("🚀 3連単の確率を予測する", type="primary", use_container_width=True):
    
    # 各艇の各着順スコア（予測確率）を取得
    boat_scores = {}
    for boat in boats_data:
        f = boat['frame']
        df_in = boat['input_df']
        p1 = models['1着率'].predict_proba(df_in)[0][1]
        p2 = models['2着以内（2連対率）'].predict_proba(df_in)[0][1]
        p3 = models['3着以内（3連対率）'].predict_proba(df_in)[0][1]
        
        boat_scores[f] = {'p1': p1, 'p2': p2, 'p3': p3}

    # 全120通りの組み合わせ（順列）に対して生スコアを計算
    combos = list(itertools.permutations(range(1, 7), 3))
    raw_results = []
    
    total_raw_score = 0.0
    
    for c in combos:
        first, second, third = c
        score = boat_scores[first]['p1'] * boat_scores[second]['p2'] * boat_scores[third]['p3']
        total_raw_score += score
        
        raw_results.append({
            '出目': f"{first}-{second}-{third}",
            'raw_score': score
        })
    
    # 全体の合計が100%になるように正規化（確率化）
    results_df = pd.DataFrame(raw_results)
    results_df['確率'] = (results_df['raw_score'] / total_raw_score) * 100
    results_df = results_df.sort_values(by='確率', ascending=False).reset_index(drop=True)
    results_df['順位'] = results_df.index + 1

    # --------------------------------------------------
    # 3. 結果の表示
    # --------------------------------------------------
    st.markdown("---")
    st.header("📊 3連単 予測確率ランキング")
    
    # TOP 5の表示
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
            
    # 全120通りのテーブル表示
    st.subheader("📋 全120通り 確率一覧（高順位順）")
    
    display_df = results_df[['順位', '出目', '確率']].copy()
    display_df['確率'] = display_df['確率'].map('{:.2f}%'.format)
    
    st.dataframe(display_df, use_container_width=True, height=400)
