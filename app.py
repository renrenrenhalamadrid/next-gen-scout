import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. 設定 & デザイン
# ==========================================
st.set_page_config(page_title="Next Gen Scout Pro", page_icon="⚽", layout="wide")

# デザインCSS
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    div.stButton > button {
        background-color: #000000; color: #ffffff; border: 1px solid #000000; font-weight: bold; width: 100%;
    }
    div.stButton > button:hover {
        background-color: #333333; color: #ffffff; border-color: #333333;
    }
    h1, h2, h3 { color: #000000 !important; font-family: 'Helvetica', 'Arial', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .streamlit-expanderHeader { background-color: #ffffff; color: #000000; border: 1px solid #000000; }
    
    .roi-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; color: white; display: inline-block; margin-bottom: 5px; }
    .rank-s { background-color: #000000; border: 1px solid gold; color: gold; }
    .rank-a { background-color: #333333; color: white; }
    .rank-b { background-color: #cccccc; color: black; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Next Gen Scout Pro")

# ==========================================
# 2. 関数定義
# ==========================================
def format_currency(value):
    if value >= 1000000: return f"€{value/1000000:.1f}M"
    elif value >= 1000: return f"€{value/1000:.0f}k"
    else: return f"€{value}"

FOOT_MAPPING = {'right': '右足', 'left': '左足', 'both': '両足', 'nan': '不明'}
def format_foot(val): return FOOT_MAPPING.get(str(val).lower(), str(val))

def get_roi_badge(score, all_scores):
    if score >= np.percentile(all_scores, 95): return f'<span class="roi-badge rank-s">💎 Sランク: 神コスパ</span>'
    elif score >= np.percentile(all_scores, 80): return f'<span class="roi-badge rank-a">💰 Aランク: お買い得</span>'
    else: return f'<span class="roi-badge rank-b">😐 Bランク: 適正</span>'

# ==========================================
# 3. データ読み込み
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/players_with_stats.csv')
        season_df = pd.read_csv('data/season_stats.csv')
    except FileNotFoundError: return pd.DataFrame(), pd.DataFrame()

    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    today = datetime.now()
    df['age'] = (today - df['date_of_birth']).dt.days // 365
    
    # ★修正: current_club_name を読み込むように追加
    features = ['player_id', 'name', 'current_club_name', 'age', 'height_in_cm', 'position', 'market_value_in_eur', 
                'country_of_citizenship', 'goals', 'assists', 'minutes_played', 'foot', 'matches']
    
    # カラムが存在するか確認してからフィルタリング（エラー防止）
    available_features = [col for col in features if col in df.columns]
    df = df[available_features].dropna().reset_index(drop=True)
    
    safe_matches = df['matches'].replace(0, 1)
    df['goals_per_match'] = df['goals'] / safe_matches
    
    safe_value = df['market_value_in_eur'].replace(0, 100000) / 1000000
    df['roi_score'] = (df['goals'] + df['assists']) / safe_value
    
    return df, season_df

df, season_df = load_data()

if df.empty:
    st.error("❌ データが見つかりません！")
    st.stop()

# ==========================================
# 4. サイドバー
# ==========================================
st.sidebar.title("MENU")
mode = st.sidebar.radio("モード選択", ["🔍 類似選手スカウト", "💎 お買い得発掘ランキング"])

st.sidebar.markdown("---")
st.sidebar.header("共通条件")
budget_range = st.sidebar.slider("予算範囲 (€)", 0, 150000000, (0, 50000000), step=500000, format="€%d")
min_budget, max_budget = budget_range
age_range = st.sidebar.slider("年齢の範囲", 15, 45, (16, 35))
min_age, max_age = age_range
all_countries = sorted(df['country_of_citizenship'].unique())
selected_countries = st.sidebar.multiselect("国籍で絞り込む", all_countries)

if 'search_results' not in st.session_state: st.session_state['search_results'] = None
if 'target_player' not in st.session_state: st.session_state['target_player'] = None


# ==========================================
# モードA: 類似選手スカウト
# ==========================================
if mode == "🔍 類似選手スカウト":
    st.sidebar.header("ターゲット設定")
    player_name_input = st.sidebar.text_input("目標選手名（英語）", "Mitoma")
    
    if st.sidebar.button("スカウト開始"):
        target = df[df['name'].str.contains(player_name_input, case=False)]
        if len(target) == 0:
            st.error(f"選手 '{player_name_input}' が見つかりませんでした。")
            st.session_state['search_results'] = None
        else:
            target = target.iloc[0]
            st.session_state['target_player'] = target
            
            candidates = df[df['position'] == target['position']].copy()
            candidates = candidates[
                (candidates['market_value_in_eur'] >= min_budget) & 
                (candidates['market_value_in_eur'] <= max_budget) &
                (candidates['age'] >= min_age) & (candidates['age'] <= max_age)
            ]
            if selected_countries:
                candidates = candidates[candidates['country_of_citizenship'].isin(selected_countries)]

            if len(candidates) == 0:
                st.warning("条件に合う選手がいませんでした。")
                st.session_state['search_results'] = None
            else:
                feature_cols = ['age', 'height_in_cm', 'market_value_in_eur', 'goals', 'assists']
                X = candidates[feature_cols].values
                target_vec = target[feature_cols].values.reshape(1, -1)
                candidates['similarity'] = cosine_similarity(X, target_vec)
                candidates = candidates.sort_values(by='similarity', ascending=False)
                candidates = candidates[candidates['name'] != target['name']]
                st.session_state['search_results'] = candidates

    if st.session_state['search_results'] is not None:
        target = st.session_state['target_player']
        recommendations = st.session_state['search_results'].head(5)
        
        # 1. ターゲット情報
        with st.container():
            target_badge = get_roi_badge(target['roi_score'], df['roi_score'])
            # ★クラブ名を表示
            club_name = target['current_club_name'] if 'current_club_name' in target else "Unknown"
            
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 20px; border: 2px solid #000000; margin-bottom: 20px; box-shadow: 5px 5px 0px #cccccc;">
                <h2 style="margin:0; color:#000;">🎯 {target['name']} <span style="font-size: 0.6em; color: #555;">({club_name})</span></h2>
                <div style="margin-top: 10px;">{target_badge} (ROI: {target['roi_score']:.2f})</div>
                <div style="display: flex; gap: 20px; margin-top: 15px; color: #333;">
                    <div><b>年齢:</b> {target['age']}</div>
                    <div><b>身長:</b> {target['height_in_cm']}cm</div>
                    <div><b>利き足:</b> {format_foot(target['foot'])}</div>
                    <div><b>市場価値:</b> {format_currency(target['market_value_in_eur'])}</div>
                    <div><b>G/A:</b> {int(target['goals'])}G / {int(target['assists'])}A</div>
                </div>
            </div>""", unsafe_allow_html=True)
        
        # 2. ターゲットのグラフ & 履歴 (復活！)
        st.subheader(f"📈 {target['name']} のシーズン詳細")
        target_season = season_df[season_df['player_id'] == target['player_id']].sort_values('season')
        
        if not target_season.empty:
            fig_line = px.line(target_season, x='season', y=['goals', 'assists'], markers=True, hover_data=['club_name', 'matches'],
                               labels={'season': 'シーズン', 'value': '数'}, color_discrete_sequence=['#000000', '#888888'])
            new_names = {'goals': 'ゴール', 'assists': 'アシスト'}
            fig_line.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
            st.plotly_chart(fig_line, use_container_width=True)
            
            with st.expander(f"📅 {target['name']} のシーズン別成績表を見る", expanded=True):
                st.dataframe(target_season[['season', 'club_name', 'matches', 'goals', 'assists']].sort_values('season', ascending=False), hide_index=True, use_container_width=True)

        # 3. マネーボール
        st.write("---")
        st.subheader("📊 マネー・ボール分析")
        scatter_data = st.session_state['search_results'].head(50)
        fig_scatter = px.scatter(
            scatter_data, x="market_value_in_eur", y="goals", color="age", size="matches",
            hover_name="name", text="name", height=500, color_continuous_scale='Greys',
            labels={"market_value_in_eur": "市場価値 (€)", "goals": "通算ゴール"}
        )
        fig_scatter.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 4. Head-to-Head
        st.write("---")
        st.header("⚖️ Head-to-Head: 徹底比較")
        candidate_names = recommendations['name'].tolist()
        selected_rival_name = st.selectbox("詳細比較する選手を選択", candidate_names, key="rival_select")
        rival = recommendations[recommendations['name'] == selected_rival_name].iloc[0]
        
        h_col1, h_col2, h_col3 = st.columns([1, 1, 2])
        with h_col1:
            st.info("ターゲット")
            st.markdown(f"**{target['name']}**<br>{target['age']}歳 / {target['height_in_cm']}cm<br>{format_currency(target['market_value_in_eur'])}<br>{int(target['goals'])}G / {int(target['assists'])}A<br>⚡ {target['goals_per_match']:.2f} G/M", unsafe_allow_html=True)
        with h_col2:
            st.success("候補者")
            price_arrow = "💰" if target['market_value_in_eur'] > rival['market_value_in_eur'] else ""
            gpm_arrow = "🔥" if rival['goals_per_match'] > target['goals_per_match'] else ""
            st.markdown(f"**{rival['name']}**<br>{rival['age']}歳<br>{format_currency(rival['market_value_in_eur'])} {price_arrow}<br>{int(rival['goals'])}G / {int(rival['assists'])}A<br>⚡ {rival['goals_per_match']:.2f} G/M {gpm_arrow}", unsafe_allow_html=True)
        with h_col3:
            comp_data = pd.DataFrame({
                'Stats': ['年齢', '身長', 'ゴール', 'アシスト'],
                target['name']: [target['age'], target['height_in_cm'], target['goals'], target['assists']],
                rival['name']: [rival['age'], rival['height_in_cm'], rival['goals'], rival['assists']]
            })
            comp_long = comp_data.melt(id_vars='Stats', var_name='Player', value_name='Value')
            fig_comp = px.bar(comp_long, x='Stats', y='Value', color='Player', barmode='group', height=200, color_discrete_sequence=['#333333', '#999999'])
            st.plotly_chart(fig_comp, use_container_width=True)
            
            gpm_data = pd.DataFrame({'Player': [target['name'], rival['name']], 'G/M': [target['goals_per_match'], rival['goals_per_match']]})
            fig_gpm = px.bar(gpm_data, x='Player', y='G/M', color='Player', height=200, text='G/M', title="決定力 (Goals Per Match)", color_discrete_sequence=['#333333', '#999999'])
            fig_gpm.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_gpm, use_container_width=True)

        # 5. 詳細リスト
        st.write("---")
        st.subheader(f"🎯 おすすめ選手 Top 5 詳細")
        for index, row in recommendations.iterrows():
            highlight = "👈 Check!" if row['name'] == selected_rival_name else ""
            badge_html = get_roi_badge(row['roi_score'], df['roi_score'])
            # ★クラブ名表示
            cand_club = row['current_club_name'] if 'current_club_name' in row else "Unknown"
            
            with st.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    # 名前とクラブ名を併記
                    st.subheader(f"🏃 {row['name']} {highlight}")
                    st.write(f"🏠 **{cand_club}**")
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.write(f"💰 市場価値: **{format_currency(row['market_value_in_eur'])}**")
                    st.write(f"📊 通算: {int(row['matches'])}試合 / {int(row['goals'])}G / {int(row['assists'])}A")
                    st.write(f"⚡ 決定力: **{row['goals_per_match']:.2f} G/M**")
                    st.write(f"AI類似度: {round(row['similarity']*100, 1)}%")
                    
                    # ★シーズン詳細履歴（グラフ＋表）復活！
                    with st.expander("📅 詳細データ（シーズン履歴）を見る"):
                        player_season = season_df[season_df['player_id'] == row['player_id']].sort_values('season')
                        if not player_season.empty:
                            fig_cand = px.line(player_season, x='season', y=['goals', 'assists'], markers=True, height=200, color_discrete_sequence=['#000000', '#888888'])
                            st.plotly_chart(fig_cand, use_container_width=True)
                            st.dataframe(player_season[['season', 'club_name', 'matches', 'goals', 'assists']].sort_values('season', ascending=False), hide_index=True)
                        else:
                            st.write("詳細データがありません")
                
                with col2:
                    # レーダーチャート
                    goal_score = min(100, row['goals'] * 2)
                    assist_score = min(100, row['assists'] * 3.3)
                    youth_score = max(0, min(100, (40 - row['age']) * 4))
                    value_score = max(0, min(100, (1 - (row['market_value_in_eur'] / max_budget)) * 100))
                    height_score = max(0, min(100, (row['height_in_cm'] - 160) * 2.5))
                    
                    categories = ['決定力', 'アシスト', '若さ', 'コスパ', 'フィジカル']
                    values = [goal_score, assist_score, youth_score, value_score, height_score]
                    values += values[:1]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values, theta=categories, fill='toself', name=row['name'],
                        line=dict(color='black'), fillcolor='rgba(0, 0, 0, 0.2)'
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(t=20, b=20, l=40, r=40))
                    st.plotly_chart(fig, use_container_width=True)
            st.divider()


# ==========================================
# モードB: お買い得発掘ランキング
# ==========================================
elif mode == "💎 お買い得発掘ランキング":
    st.sidebar.header("ランキング設定")
    positions = sorted(df['position'].unique())
    selected_position = st.sidebar.selectbox("ポジションを選択", positions)
    
    if st.sidebar.button("ランキング作成"):
        filtered_df = df[
            (df['position'] == selected_position) &
            (df['market_value_in_eur'] >= min_budget) &
            (df['market_value_in_eur'] <= max_budget) &
            (df['age'] >= min_age) & (df['age'] <= max_age)
        ].copy()
        
        if selected_countries:
            filtered_df = filtered_df[filtered_df['country_of_citizenship'].isin(selected_countries)]
            
        ranked_df = filtered_df.sort_values(by='roi_score', ascending=False).head(20)
        
        if len(ranked_df) == 0:
            st.warning(f"条件に合う選手がいませんでした。\n予算: {format_currency(min_budget)} - {format_currency(max_budget)}")
        else:
            st.subheader(f"💎 {selected_position} のお買い得選手ランキング (Top 20)")
            st.caption(f"予算: {format_currency(min_budget)}-{format_currency(max_budget)} / 年齢: {min_age}-{max_age}歳 / 地域: {selected_countries if selected_countries else 'All'}")
            
            for i, (index, row) in enumerate(ranked_df.iterrows()):
                rank = i + 1
                badge_html = get_roi_badge(row['roi_score'], df['roi_score'])
                # ★クラブ名
                club_name = row['current_club_name'] if 'current_club_name' in row else "Unknown"
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.markdown(f"<h1 style='text-align: center; color: #333;'>#{rank}</h1>", unsafe_allow_html=True)
                    with col2:
                        st.subheader(f"{row['name']}")
                        st.write(f"🏠 **{club_name}**") # ★クラブ名表示
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.write(f"国籍: {row['country_of_citizenship']} / 年齢: {row['age']}歳")
                    with col3:
                        st.metric("市場価値", format_currency(row['market_value_in_eur']))
                        st.metric("ROIスコア", f"{row['roi_score']:.2f}", delta="コスパ指数")
                        
                    with st.expander("詳細データ（シーズン履歴）を見る"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"📊 通算成績: {int(row['goals'])}G / {int(row['assists'])}A")
                            st.write(f"⚡ 決定力: {row['goals_per_match']:.2f} G/M")
                        with c2:
                            player_season = season_df[season_df['player_id'] == row['player_id']].sort_values('season')
                            if not player_season.empty:
                                fig = px.line(player_season, x='season', y=['goals', 'assists'], markers=True, height=200, color_discrete_sequence=['#000000', '#888888'])
                                st.plotly_chart(fig, use_container_width=True)
                                # ★表もここに表示
                                st.dataframe(player_season[['season', 'club_name', 'matches', 'goals', 'assists']].sort_values('season', ascending=False), hide_index=True)
                st.markdown("---")
                