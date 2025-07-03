import streamlit as st
import re
import pickle
from sklearn.linear_model import LogisticRegression


st.header("News Detection System", divider=True)
st.text("This system has an accuracy of 83%")
st.page_link("https://www.kaggle.com/datasets/algord/fake-news", label=" Dataset", icon="📙")

# Load model
model = pickle.load(open('model.pkl', 'rb'))
# Load vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
# Load stopwords
stopwords = pickle.load(open('stopwords.pkl', 'rb'))
# Load Stemmer
ps = pickle.load(open('ps.pkl', 'rb'))




def stemming(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)   # remove punctuation and special chars
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # remove extra spaces
    words = sentence.split()                          # split into words
    filtered = [ps.stem(word) for word in words if word not in stopwords]
    return ' '.join(filtered)



def main():
    title = st.text_area("Enter News Headline: ",)
    st.write(title)

    predict = "False"
    
    


    if st.button("Detect", type="primary"):
        news = stemming(title) 
        vector_trans = vectorizer.transform([news])
        prediction_test = model.predict(vector_trans)
        if(prediction_test == 0):
            predict = "FAKE"
        else:
            predict = 'REAL '

        
        st.write(f"The News Detection is: {predict}")
        st.header(predict)




main()