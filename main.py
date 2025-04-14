import random
import streamlit as st
import lmstudio as lms
import pandas as pd
import io
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import MeCab
import re
from collections import Counter
import matplotlib
from PIL import Image


# 環境設定
import os
os.environ["MECABRC"] = "/opt/homebrew/etc/mecabrc"

# MeCab設定
tagger = MeCab.Tagger("-Ochasen -d /opt/homebrew/lib/mecab/dic/ipadic")

# フォント設定
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['font.size'] = 12

# ページ設定
st.set_page_config(
    page_title="LLMアンケートシミュレーター",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("LLMによるアンケートシミュレーション")
st.markdown("様々な属性を持つバーチャルユーザーに対して、アンケート調査をシミュレーションします。")

# サイドバーでパラメータ設定
st.sidebar.header("設定")
num_agents = st.sidebar.slider("バーチャルユーザー数", 1, 10, 3)
temperature = st.sidebar.slider("多様性 (temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("最大トークン数", 100, 1000, 100000)

# LMStudioモデルの取得
@st.cache_resource
def load_model():
    return lms.llm("gemma-3-4b-it")  # デフォルトモデルを使用

model = load_model()

# 既存のターゲット属性の基本設定
default_target_attributes = {
    "年齢": [18, 25, 35, 45, 55, 65],
    "性別": ["男性", "女性", "ノンバイナリー"],
    "国籍": ["アメリカ", "日本", "イギリス", "ドイツ", "フランス", "中国"],
    "職業": ["学生", "エンジニア", "教師", "医師", "アーティスト"],
    "趣味": ["読書", "スポーツ", "ゲーム", "旅行", "料理"],
    "サービス経験": ["あり", "なし"],
    "サービス発見経路": ["広告", "友人", "ソーシャルメディア", "検索エンジン"]
}

# セッション状態の初期化
if 'target_attributes' not in st.session_state:
    st.session_state.target_attributes = default_target_attributes.copy()

if 'generated_attributes' not in st.session_state:
    st.session_state.generated_attributes = False

# LLMを使用して属性値を生成する関数
def generate_attribute_values(attribute_name, num_values=5):
    try:
        prompt = f"""以下の属性について、多様なユーザープロファイル作成のための選択肢を{num_values}個生成してください。
属性名: {attribute_name}
各選択肢は1単語または短いフレーズで、具体的かつ多様なものにしてください。
選択肢のリストのみを出力してください。"""
        
        chat = lms.Chat("あなたはユーザープロファイリングの専門家です。多様かつ具体的な属性値を生成してください。")
        chat.add_user_message(prompt)
        result = model.respond(chat, config={
            "temperature": 0.9,
            "maxTokens": 500,
            "topP": 0.9
        })
        
        # 結果をリスト化
        values = [v.strip() for v in result.content.strip().split('\n') if v.strip()]
        if len(values) < 2:  # 少なくとも2つの選択肢が必要
            return default_target_attributes.get(attribute_name, ["値が生成できませんでした"])
        return values
    except Exception as e:
        st.warning(f"属性「{attribute_name}」の値生成中にエラーが発生しました: {e}")
        return default_target_attributes.get(attribute_name, ["値が生成できませんでした"])

# セッションに保存するための関数
def generate_and_save_attributes():
    attribute_status.info("属性を生成中...")
    
    # 属性名のリスト
    attribute_names = ["年齢", "性別", "国籍", "職業", "趣味", "サービス経験", "サービス発見経路"]
    generated_attributes = {}
    
    # 各属性について値を生成
    progress = st.sidebar.progress(0)
    for i, attr in enumerate(attribute_names):
        progress.progress((i+0.5)/len(attribute_names))
        generated_attributes[attr] = generate_attribute_values(attr, num_attribute_values)
    
    progress.progress(1.0)
    
    # セッション状態に保存
    st.session_state.target_attributes = generated_attributes
    st.session_state.generated_attributes = True
    
    attribute_status.success("属性生成完了！")

# サイドバーに属性生成オプションを追加
st.sidebar.header("属性設定")
use_llm_attributes = st.sidebar.checkbox("LLMで属性を生成", value=st.session_state.generated_attributes)
num_attribute_values = st.sidebar.slider("各属性の選択肢数", 3, 10, 5)

# 生成状態を表示するプレースホルダー
attribute_status = st.sidebar.empty()

# 生成ボタンとその処理
if use_llm_attributes:
    if st.sidebar.button("属性を生成"):
        generate_and_save_attributes()
    
    # 生成された属性を表示
    if st.session_state.generated_attributes:
        with st.sidebar.expander("生成された属性", expanded=True):
            for attr, values in st.session_state.target_attributes.items():
                st.write(f"**{attr}**: {', '.join(str(v) for v in values)}")
else:
    # デフォルト値に戻す
    st.session_state.target_attributes = default_target_attributes.copy()
    st.session_state.generated_attributes = False

# アンケート調査の質問を定義
questions = [
    "新しいサービスの印象はどうですか？",
    "このサービスを利用する可能性はどのくらいありますか？",
    "最も魅力的だと思う機能は何ですか？",
    "どのような改善点を提案しますか？",
    "類似のサービスをどのように発見しましたか？"
]

# 質問セクション
st.subheader("質問設定")

# デフォルトの質問を表示
with st.expander("デフォルトの質問", expanded=True):
    for i, question in enumerate(questions):
        st.write(f"{i+1}. {question}")
    st.info("これらの質問が全ての回答者に対して使用されます。")

# セッション状態で追加質問を管理
if 'additional_questions' not in st.session_state:
    st.session_state.additional_questions = []
    
# 追加質問の表示
if st.session_state.additional_questions:
    st.subheader("追加された質問")
    for i, question in enumerate(st.session_state.additional_questions):
        st.write(f"{len(questions) + i + 1}. {question}")

# その他のセッション状態の管理
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

if 'show_success' not in st.session_state:
    st.session_state.show_success = False

# 「質問を追加」ボタンが押されたときの処理を行うコールバック関数
def add_question():
    if st.session_state.question_input:
        new_question = st.session_state.question_input
        if new_question not in questions and new_question not in st.session_state.additional_questions:
            st.session_state.additional_questions.append(new_question)
            st.session_state.show_success = True
            # フォーム送信後に入力欄をクリア
            st.session_state.question_input = ""

# 追加質問の入力フォーム
with st.form(key="add_question_form"):
    st.text_input("追加の質問を入力してください", 
                key="question_input",
                value=st.session_state.question_input)
    submit_button = st.form_submit_button("質問を追加", on_click=add_question)
    
    if submit_button and not st.session_state.question_input:
        if st.session_state.get('show_success', False):
            # ボタンが押された後のリロードで、入力が空になっていても成功フラグがある場合は何もしない
            pass
        else:
            st.warning("質問を入力してください。")
    elif submit_button and st.session_state.question_input in questions or st.session_state.question_input in st.session_state.additional_questions:
        st.warning("その質問はすでに追加されています。")

# フォーム外でメッセージを表示
if st.session_state.get('show_success', False):
    st.success(f"質問が追加されました！")
    st.session_state.show_success = False

# すべての質問を結合（デフォルト質問＋追加質問）
all_questions = questions + st.session_state.additional_questions

# 追加質問のリセットボタン
if st.session_state.additional_questions:
    if st.button("追加質問をリセット"):
        st.session_state.additional_questions = []
        st.success("追加質問がリセットされました。")
        st.rerun()

# 質問数の表示
st.info(f"合計質問数: {len(all_questions)}問")

# サービス説明のセクション
st.subheader("サービス説明")

# 説明入力方法の選択
description_method = st.radio(
    "サービス説明の入力方法を選択",
    ["テキスト入力", "ファイルアップロード"],
    horizontal=True
)

service_description = ""

if description_method == "テキスト入力":
    service_description = st.text_area(
        "アンケート対象のサービスについて説明してください", 
        "これは新しいオンライン学習プラットフォームです。様々な分野の講座を提供し、インタラクティブな学習体験を提供します。",
        height=150
    )
else:
    # ファイルアップローダー
    uploaded_file = st.file_uploader("サービス説明のファイルをアップロード", type=["txt", "pdf", "csv", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # ファイルの種類に応じて処理
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        try:
            if file_type == "txt":
                # テキストファイルの処理
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                service_description = stringio.read()
                
            elif file_type == "pdf":
                # PDFファイルの処理
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                service_description = text
                
            elif file_type == "csv":
                # CSVファイルの処理
                df = pd.read_csv(uploaded_file)
                if "description" in df.columns:
                    service_description = "\n".join(df["description"].tolist())
                else:
                    # 最初の列を使用
                    service_description = "\n".join(df.iloc[:, 0].tolist())
            
            # 画像ファイルの処理
            elif file_type in ["jpg", "jpeg", "png"]:
                try:
                    # 画像を表示
                    image = Image.open(uploaded_file)
                    st.image(image, caption="アップロードされた画像", use_container_width=True)
                    
                    # 処理状態を表示
                    processing_status = st.info("画像を解析中...")
                    
                    try:
                        # 画像をbase64エンコード
                        buffered = io.BytesIO()
                        # RGBAモードの場合はRGBに変換
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                            
                        # 画像サイズを縮小して処理速度を上げる
                        max_size = (512, 512)
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        
                        # 圧縮率を高めてサイズを削減
                        image.save(buffered, format="JPEG", quality=70)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # 適切なAPIフォーマットでメッセージを構築
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "この画像について詳しく説明してください。画像が示すサービスやプロダクトの内容、特徴、ターゲットユーザーについて具体的に解説してください。"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                                    }
                                ]
                            }
                        ]
                        
                        # LMStudioのチャットAPIを呼び出し
                        chat = lms.Chat()
                        chat.add_system_message("あなたは画像を分析して詳細な説明を提供するアシスタントです。")
                        
                        result = model.respond(chat, messages=messages, config={
                            "temperature": 0.7,
                            "maxTokens": 500,
                            "topP": 0.9
                        })
                        
                        # 結果を設定
                        service_description = result.content.strip()
                        processing_status.success("画像の解析が完了しました！")
                        
                        # 解析結果を表示
                        st.subheader("画像の解析結果")
                        st.write(service_description)
                        
                    except Exception as img_error:
                        # エラーの詳細を表示
                        error_details = str(img_error)
                        processing_status.error(f"画像の解析中にエラーが発生しました: {error_details}")
                        
                        # 代替として手動入力を提供
                        user_description = st.text_area(
                            "代わりに画像の説明を入力してください",
                            height=150
                        )
                        
                        if user_description:
                            service_description = user_description
                            st.success("説明が入力されました")
                        else:
                            service_description = "画像の解析に失敗しました。テキスト入力を使用してください。"
                
                except Exception as e:
                    st.error(f"画像の処理中にエラーが発生しました: {e}")
                    service_description = "画像の処理に失敗しました。テキスト入力を使用してください。"
            
            # 説明の長さをチェック
            if len(service_description) > 10000:
                st.warning("サービス説明が長すぎます。最初の10000文字を使用します。")
                service_description = service_description[:10000]
                
            # プレビュー表示
            with st.expander("サービス説明のプレビュー", expanded=True):
                st.write(service_description)
                st.info(f"文字数: {len(service_description)}文字")
                
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
            service_description = "ファイルからの読み込みに失敗しました。テキスト入力を使用してください。"

# サービス説明が空の場合のデフォルト値
if not service_description.strip():
    service_description = "これは新しいオンライン学習プラットフォームです。様々な分野の講座を提供し、インタラクティブな学習体験を提供します。"
    st.warning("サービス説明が空です。デフォルトの説明を使用します。")

# エージェントの特性をランダムに選択
def generate_agent_attributes():
    return {key: random.choice(value) for key, value in st.session_state.target_attributes.items()}

# LMStudioを使用してエージェントの回答を生成
def get_agent_response(agent_attributes, question, service_desc):
    prompt = f"""あなたは{agent_attributes['年齢']}歳の{agent_attributes['性別']}で、{agent_attributes['国籍']}出身です。
{agent_attributes['職業']}として働いており、{agent_attributes['趣味']}を楽しんでいます。
類似のサービスの経験は{agent_attributes['サービス経験']}で、{agent_attributes['サービス発見経路']}を通じて発見しました。

あなたは以下のサービスを実際に使用したと想定してください。
サービスの詳細情報:
------
{service_desc}
------

上記のサービスに関する以下の質問に対して、あなたの経験と属性に基づいて回答してください：
質問: {question}
回答:"""
    
    try:
        chat = lms.Chat("あなたはアンケート回答者です。与えられた属性と経験に基づいて、質問に対して具体的かつ自然な回答をしてください。")
        chat.add_user_message(prompt)
        result = model.respond(chat, config={
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": 0.9
        })
        return result.content.strip()
    except Exception as e:
        return f"エラーが発生しました: {e}"

# レポートを生成
def generate_report(survey_results):
    report = ""
    for result in survey_results:
        report += f"回答者の属性: {result['属性']}\n"
        for question, response in result['回答'].items():
            report += f"{question}\n回答: {response}\n\n"
    return report

# LLMを使って結果を要約
def summarize_results(report):
    try:
        summarize_chat = lms.Chat("あなたはアンケート結果の分析専門家です。以下のアンケート結果を要約し、重要な洞察を抽出してください。")
        summarize_chat.add_user_message(f"以下のアンケート結果を要約してください：\n\n{report}")
        summary_result = model.respond(summarize_chat, config={
            "temperature": temperature,
            "maxTokens": 1000,
            "topP": 0.9
        })
        return summary_result.content.strip()
    except Exception as e:
        return f"要約中にエラーが発生しました: {e}"

# 形態素解析による日本語の名詞・動詞・形容詞の抽出
def extract_keywords(text):
    tagger = MeCab.Tagger("-Ochasen")
    parsed = tagger.parse(text)
    keywords = []
    
    for line in parsed.split('\n'):
        if line == 'EOS' or line == '':
            continue
        parts = line.split('\t')
        if len(parts) >= 4:
            pos = parts[3]  # 品詞情報
            # 名詞、動詞、形容詞のみを抽出
            if pos.startswith('名詞') or pos.startswith('動詞') or pos.startswith('形容詞'):
                word = parts[0]  # 単語
                # 2文字以上のものだけ対象にする
                if len(word) >= 2 and not word.isdigit() and re.match(r'^[a-zA-Z0-9]+$', word) is None:
                    keywords.append(word)
    return keywords

# 実行ボタン
if st.button("アンケートを実行"):
    # 結果の表示用コンテナを準備
    results_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with results_container:
        st.subheader("アンケート結果")
        # タブで結果を整理
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["個別回答", "要約", "生データ", "ワードクラウド", "マインドマップ"])
        
        with tab1:
            # 個別回答用のプレースホルダー
            respondent_containers = []
            for i in range(num_agents):
                # 各回答者用のエクスパンダーを作成
                respondent_containers.append(
                    st.expander(f"回答者 {i+1} (生成中...)", expanded=(i==0))
                )
        
        # 要約と生データ用のプレースホルダー
        with tab2:
            summary_placeholder = st.empty()
            summary_placeholder.info("アンケート完了後に要約を生成します...")
        
        with tab3:
            raw_data_placeholder = st.empty()
            raw_data_placeholder.info("アンケート実行中...")
        
        with tab4:
            wordcloud_placeholder = st.empty()
            wordcloud_placeholder.info("アンケート完了後にワードクラウドを生成します...")
        
        with tab5:
            mindmap_placeholder = st.empty()
            mindmap_placeholder.info("アンケート完了後にマインドマップを生成します...")
    
    # 結果を格納するリスト
    survey_results = []
    
    # アンケート実施（回答者ごとに処理）
    for i in range(num_agents):
        status_text.text(f"回答者 {i+1}/{num_agents} の回答を生成中...")
        agent_attributes = generate_agent_attributes()
        agent_responses = {}
        
        # 質問ごとに処理
        for q_idx, question in enumerate(all_questions):
            response = get_agent_response(agent_attributes, question, service_description)
            agent_responses[question] = response
            progress_percent = (i * len(all_questions) + q_idx + 1) / (num_agents * len(all_questions))
            progress_bar.progress(progress_percent)
    
        # 結果をリストに追加
        survey_results.append({"属性": agent_attributes, "回答": agent_responses})
        
        # 回答者ごとの結果をリアルタイムで表示
        with respondent_containers[i]:
            st.write("### 属性")
            for key, value in agent_attributes.items():
                st.write(f"**{key}**: {value}")
            
            st.write("### 回答")
            for question, response in agent_responses.items():
                st.write(f"**質問**: {question}")
                st.write(f"**回答**: {response}")
                st.write("---")
        
        # エクスパンダーのタイトルを更新
        respondent_containers[i].header(
            f"回答者 {i+1} ({agent_attributes['職業']}, {agent_attributes['年齢']}歳, {agent_attributes['性別']})"
        )
        
        # 途中経過のレポートを更新
        current_report = generate_report(survey_results)
        raw_data_placeholder.text_area("生データ", current_report, height=500)
    
    status_text.text("アンケート完了！要約を生成中...")
    
    # すべての回答が完了したら要約を生成
    final_report = generate_report(survey_results)
    summary = summarize_results(final_report)
    summary_placeholder.markdown("### アンケート結果の要約")
    summary_placeholder.write(summary)
    
    # ワードクラウドとマインドマップの生成
    status_text.text("可視化を生成中...")
    
    # ワードクラウド生成
    with tab4:
        # すべての回答を結合したテキスト
        all_responses_text = " ".join([response for result in survey_results for response in result["回答"].values()])
        
        # ストップワード（除外する単語）
        stop_words = ['です', 'ます', 'した', 'ある', 'ない', 'いる', 'これ', 'それ', 'あり', 'なし',
                    'こと', 'もの', 'よう', 'あの', 'この', 'その', 'ので', 'から', 'ため']
        
        try:
            # WordCloudオブジェクトを生成
            wordcloud = WordCloud(
                width=1200, 
                height=800, 
                background_color='white',
                stopwords=set(stop_words),
                max_words=150,
                colormap='viridis',
                font_path='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'  # macOS用フォント
            ).generate(all_responses_text)
            
            # ワードクラウドを描画
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Streamlitに表示
            wordcloud_placeholder.subheader("回答内容のワードクラウド")
            wordcloud_placeholder.pyplot(fig)
            
        except Exception as e:
            # エラー発生時はシンプルなワードクラウドを試す
            wordcloud_placeholder.warning(f"最初のワードクラウド生成中にエラーが発生しました。シンプルな方法で再試行します。")
            
            try:
                # フォント指定なしのシンプルなワードクラウド
                simple_wordcloud = WordCloud(
                    width=1200, 
                    height=800, 
                    background_color='white',
                    max_words=150,
                    colormap='viridis'
                ).generate(all_responses_text)
                
                # 描画
                fig = plt.figure(figsize=(12, 8))
                plt.imshow(simple_wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # 表示
                wordcloud_placeholder.subheader("回答内容のワードクラウド")
                wordcloud_placeholder.pyplot(fig)
                
            except Exception as e2:
                wordcloud_placeholder.error(f"ワードクラウドの生成に失敗しました: {e2}")
                wordcloud_placeholder.info("より詳細な回答を得るには質問を追加するか、回答者数を増やしてみてください。")

    # マインドマップ生成
    with tab5:
        # 質問をノードとして追加
        G = nx.Graph()
        G.add_node("アンケート結果", size=25, color="#FF5733")
        
        # 質問ごとの処理
        for q_idx, question in enumerate(questions):
            # 質問の短縮表示用
            short_q = question[:20] + "..." if len(question) > 20 else question
            # 質問ノードを追加
            G.add_node(short_q, size=20, color="#33A8FF")
            G.add_edge("アンケート結果", short_q)
            
            # この質問に対する全回答からキーワードを抽出
            q_responses = " ".join([result["回答"][question] for result in survey_results])
            q_keywords = extract_keywords(q_responses)
            q_word_count = Counter(q_keywords)
            
            # ストップワードを除去
            for word in stop_words:
                if word in q_word_count:
                    del q_word_count[word]
            
            # 上位10個のキーワードをノードとして追加
            for word, count in q_word_count.most_common(10):
                if len(word) >= 2:  # 2文字以上の単語のみ
                    # キーワードノードを追加
                    G.add_node(word, size=10 + count, color="#B3FF33")
                    G.add_edge(short_q, word)
        
        # ネットワークの作成
        nt = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="black")
        
        # Networkxグラフをpyvisネットワークに変換
        for node in G.nodes():
            nt.add_node(node, 
                       label=node, 
                       size=G.nodes[node].get('size', 10),
                       color=G.nodes[node].get('color', "#B3FF33"))
            
        for edge in G.edges():
            nt.add_edge(edge[0], edge[1])
            
        # 物理法則の設定
        nt.barnes_hut(gravity=-10000, central_gravity=0.3, spring_length=200)
        
        # HTMLファイルとして一時保存
        path = "/tmp/mindmap.html"
        nt.save_graph(path)
        
        # HTMLファイルを読み込んでStreamlitに表示
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        mindmap_placeholder.subheader("回答内容のマインドマップ")
        st.components.v1.html(html, height=600)
        
        # マインドマップの説明
        mindmap_placeholder.info("""
        このマインドマップは質問（青）と頻出キーワード（緑）の関係を示しています。
        中心の赤いノードからアンケートの質問が枝分かれし、各質問からはそれに関連する頻出キーワードが表示されています。
        ノードの大きさは重要度（出現頻度）を表しています。
        """)
    
    status_text.text("処理完了！")

# アプリの説明
with st.expander("このアプリについて"):
    st.markdown("""
    ## LLMアンケートシミュレーターについて
    
    このアプリケーションは、LLM（大規模言語モデル）を使用して、様々な属性を持つ仮想ユーザーによるアンケート回答をシミュレーションします。
    
    ### 使い方
    1. サイドバーで回答者数やモデルのパラメータを設定
    2. 必要に応じてカスタム質問を追加
    3. アンケート対象のサービスについて説明を入力またはファイルをアップロード
    4. 「アンケートを実行」ボタンをクリックして結果を生成
    
    ### 可視化機能
    - ワードクラウド：回答全体から抽出したキーワードを視覚的に表示
    - マインドマップ：質問とキーワードの関係性を階層的に可視化
    
    ### 注意点
    - このシミュレーションは実際のユーザー調査の代わりにはなりません
    - 生成される回答はLLMの出力であり、実際の人間の意見を正確に反映するものではありません
    - 様々な属性の視点からの回答を得ることで、多角的な視点を得るための参考として活用してください
    """)