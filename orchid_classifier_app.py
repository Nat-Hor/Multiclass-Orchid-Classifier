import streamlit as st
from tensorflow import keras
from tensorflow import image as img
from tensorflow import squeeze
import matplotlib.pylab as plt
import numpy as np
import orchid_info

@st.cache()
def load_label_list() -> list:
    ds_label_list = orchid_info.orchid_namelist
    return ds_label_list

@st.cache(allow_output_mutation=True)
def load_model():
    model = orchid_model = keras.models.load_model('orchid_model1')
    return model
    
@st.cache()
def predict(
    image,
    ds_label_list: list,
    orchid_model,
) -> str:
    prediction_temp = orchid_model.predict(image)
    prediction_temp = squeeze(prediction_temp).numpy()
    prediction_id = np.argmax(prediction_temp, axis=-1)
    prediction = ds_label_list[prediction_id]
    return prediction

if __name__ == '__main__':
    # preparation for the web app function
    ds_label_list = load_label_list()
    orchid_model = load_model()
    # initialize variable
    image = np.ones((1, 224, 224, 3))
    prediction = ""

    st.title("Orchid classifier")

    description = """
        This application helps user to classify 16 orchid species that 
        can be found in Malaysia. The name of orchid species are : Dendrobium dawn maree, 
        Renanthera Kalsom, Vanda Miss Joaquim, Aerides houlletiana, 
        Brassavola nodosa, Bulbophyllum annandalei, Bulbophyllum lepibum, 
        Calanthe sylvatica, Coelogyne pandurata, Cymbidium bicolor, Eria floribunda, 
        Grammtophyllum speciosum, Paphiopedilum callosum, Phalaenopsis lowii, 
        Phaleanopsis violacea, Spathoglottis plicata. 
        Lets try it out and see how it classifies your images.
    """
    # Note that the classifier model is not 100% accurate, so it may gives wrong prediction.
    st.write(description)

    instruction = """
        \nUpload an orchid image. The image will be fed into the CNN model and the 
        output will be displayed on the screen.
        """
    st.write(instruction)
    
    file = st.file_uploader("")

    if file: 
        image = plt.imread(file)
        # preprocess the image to fit with the CNN model
        image = img.resize(image, (224, 224))
        image = np.array(image)
        image = image.reshape(1, 224, 224, 3)/255.0
        prediction = predict(image, ds_label_list, orchid_model)

    else:
        prediction = "No image file detected" 

    st.title("Your image :")
    st.image(image)
    st.title("Orchid species detected : " + prediction)
