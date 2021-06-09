import streamlit as st
import pickle


encoder_file = open("vectorizer.pkl","rb")
encoder_model = pickle.load(encoder_file)

classy = open("XGBTmodel.pkl","rb")
classifier = pickle.load(classy)





def predictSentiment(userInput):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: userInput
        in: query
        type: string
        required: true
      
    responses:
        200:
            description: The output values
        
    """
   
    HotEncode = encoder_model.transform(userInput)
    
    prediction = classifier.predict(HotEncode)

    if prediction == 1:
        return "Positive"
    elif prediction == 0 :
        return "Neutral"
    else :
        return "Negative"
        
    


def main():
    
    st.title("XGBoost based model performance on highly imbalanced dataset ")
    html_temp = """
    <div style="background-color: blue ; padding:10px">

    <h2 style="color:black;text-align:center;"> Twitter Sentiment Analysis </h2>

    <p>The deployed model is able to classify as three sentiments - positive, neutral or negative .</p>

    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    userInput = st.text_input("PLEASE TYPE THE TEXT BELOW :D")
    userInput = [userInput]
    result=""
    if st.button("Predict"):
        result=predictSentiment(userInput)

    st.success('The sentiment seams to be quite, {}'.format(result))

    if st.button("About Me"):
        st.text(" I am Samriddh Singh, a Data Science practioner.")
        st.text("I am pursuing my Bachelor education from NIT Hamirpur.")
        st.text("This simple model was deployed using Streamlit Yay !. The dataset was highly imbalanced pardon me for biased outputs on negative sentiment.  !")
        html2 = """ 
        <div style = "color : pink ; text-align:center;">
        <h1> Samriddh's contact</h1>
        <h3>
        <a href= "https://www.linkedin.com/in/samriddh-singh-70621b18b/")>[LinekdIn]</href> <br>
        <a href= "https://twitter.com/saketbyte)")>[Twitter]</href> <br>
        <a href= "https://github.com/saketbyte")>[Github]</href> <br>
        </h3>
        """
        st.markdown(html2,unsafe_allow_html=True)


if __name__ == '__main__':
    main()