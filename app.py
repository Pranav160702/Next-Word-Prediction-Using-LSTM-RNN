import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the Tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)



def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert the input text to a sequence of integers
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure the sequence length is appropriate for the model
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Keep only the last max_sequence_len - 1 tokens
    
    # Pad the sequence to match the input shape required by the model
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    
    # Find the index of the most probable next word
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Map the index to the corresponding word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    
    # Return None if no word is found
    return None


# Streamlit App
st.title("Next Word Prediction With LSTm And Early Stopping")
input_text = st.text_input("Enter the Sequence of Words:","To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word : {next_word}")

