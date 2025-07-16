import streamlit as st
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import numpy as np
from sklearn import metrics as skmetrics
from transformers import BertTokenizer, BertForSequenceClassification,T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

import re
import nltk
import string
# import sentencepiece
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import SnowballStemmer
# from symspellpy import SymSpell, Verbosity
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

import matplotlib.pyplot as plt
import seaborn as sns

class T5Model(torch.nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        # Replace the last layer with a linear layer and output 6 categories
        self.classifier = torch.nn.Linear(self.t5_model.config.d_model, 64)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Take the output of the encoder as the input of the classifier
        encoder_output = outputs.last_hidden_state.mean(dim=1)  
        logits = self.classifier(encoder_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 64), labels.view(-1))
            return loss, logits
        return logits

@st.cache_resource
def load_model(model_type):
    match model_type:
        case 'LSTM_for_binary':
            model = tf.keras.models.load_model('./models/lstm_model_sigmoid.keras')
            return model
        case 'LSTM_for_multiple':
            model = tf.keras.models.load_model('./models/lstm_model_multipleclassifier.keras')
            return model
        case 'BERT_for_binary':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            model.load_state_dict(torch.load('./models/bert_model.pt', map_location="cuda:0"))
            model=model.to(device)
            model.eval()
            return model,tokenizer
        case 'T5_for_multiple':
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            model = T5Model()
            model.load_state_dict(torch.load('./models/t5_update_model.pth', map_location="cuda:0"))
            model=model.to(device)
            model.eval()
            return model,tokenizer
        case _:
            model = None
            st.write('invalid model')

@st.cache_resource
def init():
    stemmer = PorterStemmer() 
    lem = WordNetLemmatizer()
    return stemmer,lem

def clean_string(text, stem="None"):

    # Final String to return
    final_string = ""

    # Make the text to be lower case
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', ' ', text)

    # Remove punctuations
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':     
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string


# prediction function used by LSTM
def predict_text_binary(text, model):
    # Clean the text    
    # Convert to TensorFlow tensor with dtype=tf.string
    input_text = tf.constant(text, dtype=tf.string)
    # Make prediction
    prediction = model.predict(input_text)
    
    # Convert prediction to binary
    binary_prediction = np.round(prediction).astype(int).flatten()
    
    return binary_prediction

# prediction function used by LSTM
def predict_text_multiple(text, model,threshold=0.5):
    # Clean the text    
    # Convert to TensorFlow tensor with dtype=tf.string
    input_text = tf.constant(text, dtype=tf.string)
    # Make prediction
    prediction = model.predict(input_text)
    
     # Convert probabilities to binary predictions using the threshold
    binary_prediction = (prediction >= threshold).astype(int)
    
    # Map binary predictions to labels
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predicted_labels = []
    for i, label in enumerate(label_columns):
        if binary_prediction[0][i] == 1:  # Check if the label is predicted as positive
            predicted_labels.append(label)
    
    # Return the list of predicted labels
    return predicted_labels

#predict the label of a single sentence
@st.cache_data
def predict_single_comment(text, class_num, selected_model, threshold=0.5):
    text = clean_string(text,stem='Lem')
    label= ["(Prediction failed)"]
    if selected_model == "LSTM":
        if class_num == "Binary":
            model = load_model('LSTM_for_binary')
            if model is not None:   
                predicted_label= predict_text_binary([text],model)
                label="toxic" if predicted_label[0]==1 else "non-toxic"
            else: st.write("load model error")
        else:
            model = load_model('LSTM_for_multiple')
            if model is not None:
                label = predict_text_multiple([text], model, threshold)
            else: st.write("load model error")
    elif selected_model == "BERT":
        model,tokenizer = load_model('BERT_for_binary')
        text_tokenized = tokenizer([text], truncation=True, padding='max_length', max_length=128)
        #print(text_tokenized)
        input_ids = torch.tensor(text_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(text_tokenized['attention_mask']).to(device)
        #print(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        #print("preds:",preds)
        label="toxic" if preds[0]==1 else "non-toxic"
    elif selected_model == "T5":
        model,tokenizer = load_model('T5_for_multiple')
        text_tokenized = tokenizer([text], truncation=True, padding='max_length', max_length=250, return_tensors='pt')
        #print(text_tokenized)
        label_tokenized = tokenizer(["none"], truncation=True, padding='max_length', max_length=10, return_tensors='pt')
        input_ids = text_tokenized['input_ids'].to(device)
        attention_mask = text_tokenized['attention_mask'].to(device)
        #labels = label_tokenized['input_ids'].to(device)
        labels = torch.tensor(0).to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        #print(outputs.logits.size())
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        #st.write(preds)
        label=[]
        labels=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for i in range(6):
            if preds[0] & (1<<i):
                label.append(labels[i])
        return label
    else:
        st.write("Model missed")
    return label

#remove duplicates in dataset
def remove_duplicates(df, text_column):
    df_cleaned = df.drop_duplicates(subset=[text_column], keep='first').reset_index(drop=True)
    return df_cleaned

st.title('Toxic Comment Classification')
device = torch.device("cuda")

stemmer,lem = init()

class_num = st.radio(
    "Multiple or binary classifier?",
    ["Multiple", "Binary"],
    captions=[
        "toxic, severe_toxic, obscene, threat, insult, identity_hate",
        "toxic, non-toxic",
    ],
)

if class_num == "Multiple":
    selected_model = st.selectbox(
        "Which model do you want to use?",
        ("LSTM", "T5"),
    )
else:
    selected_model = st.selectbox(
        "Which model do you want to use?",
        ("LSTM", "BERT"),
    )

#classification on a single sentence
text = st.text_input("The comment you want to classify", "type your comment here")
if class_num == "Multiple":
    if selected_model == "LSTM":
        threshold = st.slider("Prob threshold", 0.0, 1.0, value=0.5, step=0.05)
    else:
        threshold=0.5
    label = predict_single_comment(text,class_num,selected_model,threshold)
    if len(label)==0: st.write("This comment is *non-toxic*")
    else :
        label = ["*"+x+"*" for x in label]
        st.write("This comment is ",','.join(label))
else: 
    st.write("This comment is *"+predict_single_comment(text,class_num,selected_model,0.5)+"*")

#classification on uploaded data
if class_num == "Binary":
    uploaded_file = st.file_uploader('Please submit your test data. The data should be in csv format and include raw text as *comments* and \'toxic\' or \'non-toxic\' as *labels*', type=['csv'])
    if uploaded_file is not None:
        df= pd.read_csv(uploaded_file)
        df= remove_duplicates(df, 'comments')
        #df_combined['label'] = df_combined['label'].replace({1: "toxic", 0:"non_toxic"})
        'Starting a long computation...'
        latest_iteration = st.empty()
        bar = st.progress(0.0)
        predictions = []
        for i,text in enumerate(df['comments']):
            predictions.append(predict_single_comment(text,class_num,selected_model,0.5))
            latest_iteration.text(f'Processing NO.{i+1} comment')
            bar.progress((i+1)*1.0/len(df))
        '...and now we\'re done!'
        #st.write(df['labels'],predictions)
        df.replace({'toxic':1,'non-toxic':0},inplace=True)
        predictions = [ 1 if x=='toxic' else 0 for x in predictions]
        matrix = skmetrics.confusion_matrix(df['labels'],predictions)
        st.write("The confusion matrix is:")
        fig,ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=['Not Toxic', 'Toxic'], yticklabels=['Not Toxic', 'Toxic'],ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        #st.write(matrix)
        #st.table(pd.DataFrame(matrix,columns=('predicted False','predicted True')))
        st.write("accuracy:",skmetrics.accuracy_score(df['labels'],predictions))
        st.write("recall:",skmetrics.recall_score(df['labels'],predictions))
        st.write("precision:",skmetrics.precision_score(df['labels'],predictions))
    else: st.write("no data!")