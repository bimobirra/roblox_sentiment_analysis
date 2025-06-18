import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time

model = keras.models.load_model('model/lstm_model.h5', compile=False)
tokenizer = joblib.load('model/tokenizer.pkl')

st.header('Roblox Playstore Reviews Sentiment Analysis (Bahasa Indonesia)')
st.write('---')

tab1, tab2 = st.tabs(['Problem', 'Sentiment Analysis'])

with tab1:
    st.subheader('Problem')
    st.write("In this modern time, user reviews on platforms such as Google Play Store are very valueable to the developer and stakeholders alike. They provides direct insights into crucial areas for app development such as bugs, glitches, inspirations, and improvements.")
    st.write("For popular app such as Roblox, the sheer volume of daily reviews can be overwhelming, often numbering in thousand. Manually categorizing each review is not only time-consuming but also requires significant manual effort")
    st.write('---')

    st.subheader('Solution')
    st.write('To address this problem, this project implements a Machine Learning solution leveraging Natural Language Processing (NLP), by using Long-Short Term Memory (LSTM) model, this model can automatically classify reviews into positive, neutral, or negative segments. This led the developers and stakeholders to quickly understand customer feedback, and make data-driven decision.')
    st.write('---')

    st.subheader('Dataset')
    st.write("""
             The dataset for this project consists of user reviews for the Roblox app on the Google Play Store, collected using the google_play_scraper library with a focus on reviews in Bahasa Indonesia.

            Since the raw data did not have explicit sentiment labels, a programmatic labeling approach was employed. Each review was processed using an Indonesian sentiment lexicon (a predefined dictionary of positive and negative words). Based on the prevalence of these words, each review was automatically assigned a polarity label: 'positive', 'negative', or 'neutral'.

            This newly labeled dataset was then used as the ground truth to train and validate a Long Short-Term Memory (LSTM) model. The goal was to teach the model to understand context and sentiment patterns that go beyond simple keyword matching, leveraging the initial labels created by the lexicon
            """)
    st.write('---')
    

with tab2:
    
    st.subheader('Try our LSTM Model (Bahasa Indonesia only)')
    
    # Input
    input = st.text_area(
        label='Enter text:',
        height=200,
        placeholder="Example: game ini sangat bagus"
    )

    # Preidct button
    predict_button = st.button(
                    "Analyze"
                )

    if predict_button:
        # Predict Sentiment
        text_tokenize = tokenizer.texts_to_sequences([input])
        pad = pad_sequences(text_tokenize)
        
        predictions = model.predict(pad)
        
        labels = ['negative', 'positive', 'neutral']
        predicted_labels = np.argmax(predictions, axis=1)
        
        with st.spinner():
            time.sleep(2)
        
        predicted_index = predicted_labels[0]
        final_label = labels[predicted_index]
        
        st.write('---')
        st.subheader('Result:')
        
        if final_label == 'positive':
            st.success('Positive Sentiment Detected')
        elif final_label == 'negative':
            st.error('Negative Sentiment Detected')
        else:
            st.info("Neutral Sentiment Detected")