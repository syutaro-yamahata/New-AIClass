import streamlit as st
from openai import AzureOpenAI

# Azure OpenAI APIに直接アクセスするための情報

api_key = st.text_input('api_key')
endpoint = st.text_input('endpoint')
# azureにアクセスできる状態にする
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version='2024-02-01'
)

st.title('はじめてのstreamlit')

# 返答のランダム性を変えるため
temperature = st.slider('Temperature', 0.0, 1.0, step=0.1)

prompt = st.text_area('プロンプトの入力')

output_button = st.button('GPTの出力')

if output_button:
    # GPT3.5にアクセス
    response = client.chat.completions.create(
    model="gpt-35-turbo-0613", #なんのモデル？
    messages=[
        {"role":"user",
        "content":prompt}], # GPTへの入力テキスト
    temperature=temperature
    )

    st.write(response.choices[0].message.content)
