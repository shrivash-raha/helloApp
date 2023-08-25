import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
import plotly.express as px
import openai
import tiktoken

import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])


openai.api_key = ""


# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def main():
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Deloitte.svg/1920px-Deloitte.svg.png', width = 150)
    st.title("Customer Reviews Analysis")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
            
            if 'Product' in df.columns and 'Review Date' in df.columns and 'Review Title' in df.columns and 'Review' in df.columns:
                st.success("Data loaded successfully!")
                
                col1, col2, col3, col4 = st.columns([3,4, 4, 3])
                
                if col1.button("Show Data"):
                    st.write(df)
                
                # if col4.button("Perform Analysis"):
                # df = gptSentimentApi(df)
                df = analyze_reviews(df)
                st.write("Analysis complete!")
                st.write(df)
                
                col1, col2, col3, col4 = st.columns([3, 4, 4, 3])

                selected_product = col1.selectbox("Select a Product", df['Product'].unique())
                filtered_df = df[df['Product'] == selected_product]
                                    
                time_period = col4.radio("Select time period:", ["Month", "Year"])

                if time_period == "Month":
                    filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
                    filtered_df['Time Period'] = filtered_df['Review Date'].dt.to_period('M').astype('str')
                elif time_period == 'Year':
                    filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
                    filtered_df['Time Period'] = filtered_df['Review Date'].dt.to_period('Y').astype('str')
                    
                plot_sentiment_graph(filtered_df)

            else:
                st.error("Incorrect columns in the uploaded file. Required columns: 'Product', 'Review Date', 'Review Title', 'Review'")
            

        
        except Exception as e:
            st.error(f"Error loading data: {e}")



def gptSentimentApi(df):
    
# =============================================================================
    encoding = tiktoken.get_encoding("cl100k_base")

    df['tokens']=df['Review'].apply(lambda x:len(encoding.encode(x)))
    df['tokensSum'] = df['tokens'].cumsum()
    df['tokenIterNum'] = df['tokensSum']//3500
    
    dfSent = pd.DataFrame()
    for i in df.tokenIterNum.unique():
        dfIter = df[df.tokenIterNum == i].reset_index(drop = True)
        df = df[df.tokenIterNum >= i]

# =============================================================================
        review_list = dfIter['Review'].tolist()
        review_str=get_input_reviews(review_list,len(dfIter))
        prompt="Provide a sentiment score for each of these reviews:\n" + review_str + "Give the output in a scale of 0-10 against the review text."
        
        Final_response=get_response(prompt)
        Final_response = Final_response[-len(dfIter):]
    
        dfIter['Sentiment Score'] = pd.Series([i[1] for i in pd.Series(Final_response).str.split('. ')]).astype(int)
    
        dfSent = pd.concat([dfSent, dfIter]).drop(['tokens', 'tokensSum', 'tokenIterNum'], 1)

    return dfSent.reset_index(drop = True)




def get_input_reviews(review_list,end):
    review_str = ''
    for i, review in enumerate(review_list, start=1):
        tweet_str = str(i) + '. "' + review + '"\n'
        review_str += tweet_str
        if i == end:
            break
    return review_str

          
def get_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=120,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return (response["choices"][0]["text"]).split('\n')
        
        
        
def plot_sentiment_graph(df):
    graphData = df.groupby('Time Period', as_index = False)['Sentiment Score'].mean()

    fig = px.line(graphData, x='Time Period', y='Sentiment Score', title='Average Sentiment Polarity Over Time')
    fig.update_traces(mode='lines+markers', line_shape='spline')

    st.plotly_chart(fig)


def analyze_reviews(df):
    def sentiment_analysis(text):
        if isinstance(text, str):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            # subjectivity = analysis.sentiment.subjectivity
            return polarity#, subjectivity
        else:
            return None#, None
    
    def named_entity_recognition(text):
        if isinstance(text, str):
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]
            return ", ".join(entities)
        else:
            return ""

    # df["Sentiment"] = df["Review"].apply(sentiment_analysis)
    df["Sentiment Score"] = df["Review"].apply(sentiment_analysis).apply(pd.Series)
    df["Entities"] = df["Review"].apply(named_entity_recognition)
    
    return df

st.set_page_config(page_title='Customer Reviews Analysis', page_icon='https://upload.wikimedia.org/wikipedia/commons/2/2b/DeloitteNewSmall.png', layout="centered", initial_sidebar_state="auto", menu_items=None)
main()
