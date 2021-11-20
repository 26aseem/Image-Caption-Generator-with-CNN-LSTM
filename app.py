# Import the required packages
import numpy as np
import argparse
import streamlit as st
import pickle
from pickle import load
from PIL import Image

# For Model training and testing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model

# For text-to-speech
from gtts import gTTS
from playsound import playsound


### Module Definitions

# Extract features from the image
def extract_features(filename, model):    
        try:
            image = Image.open(filename)            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature


def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


# Generate Caption for the image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    text = in_text[6:-3].capitalize()
    return text



### Streamlit code starts here    
st.title("Image Caption Generator using CNN and LSTM")

# Some CSS Markdown for styling
STYLE = """
<style>
img {
     max-width: 100%;     
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# Upload an Image
format_allowed = ["JPG", "PNG", "JPEG"]
file = st.file_uploader("Upload an image to generate the caption", type=format_allowed)
show_file = st.empty()
if not file:
    show_file.info("Please upload an image in {} format".format(', '.join(format_allowed)))

else:
    # Read the image
    content = file.getvalue()
    #if isinstance(file, BytesIO):
    show_file.image(file)
    
if st.button('Generate Caption'):
    # query
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(file, xception_model)

    # Generate Description here
    description = generate_desc(model, tokenizer, photo, max_length)
    
    # Print Generated Caption
    st.title(str(description))
    
    # Language in which you want to convert
    language = 'en'
  
    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    sound_desc = gTTS(text=str(description), lang=language, slow=False)
    
    # Saving the converted audio in a mp3 file named welcome 
    sound_desc.save("description.mp3")
  
    # Playing the converted file
    playsound("./description.mp3")

    