# for web
import streamlit as st
import requests
import json
import pydeck as pdk
# general import
import numpy as np
import pandas as pd
from PIL import Image
#for nlp
import nltk
import re
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

st.set_page_config(
    page_title="JoFi - Find Your Journey is Never This Easy!",
    page_icon="✈️",
    initial_sidebar_state="auto",
    layout = "wide",
    menu_items={
        'Get Help': 'https://www.google.com/',
        'Report a bug': "https://github.com/marwanmusa",
        'About': "# JoFi - Journey Finder Application"
    }
)

st.sidebar.title('MENU')
selected = st.sidebar.radio('Select Page:',['JoFi Home','Jofi Traffic-Sign Classifier'])

if selected== 'JoFi Home':
    
    # data
    data = pd.read_csv('top_5_recomendation_with_pic.csv')
    datax = data[['latitude', 'longitude','name', 'target']]

    # banner
    image = Image.open('banner.png')
    st.image(image, use_column_width='auto', caption='Find your journey!')

    st.markdown("")
    st.markdown("")
    st.markdown("")

    # ----------------------------------------------------------------------------------------------------- #
    # image input input #
    st.markdown("<h2 style='text-align: center; color: black;'>Where Do You Want to Go?🏝️</h2>", unsafe_allow_html=True)
    # st.markdown("### ***Where do you want to go?*** 🏝️")
    uploaded_file = st.file_uploader("Upload the scenery of your destination")

    submitted = st.button('submit image')

    # assign class labels
    class_names = ['Buildings','Forest','Mountain','Sea','Street']

    # assign image size
    IMAGE_SIZE = (150,150)

    def load_prep_img(filename):
        pred = np.array(filename)[:, :, :3]
        pred = tf.image.resize(pred, size = IMAGE_SIZE)
        pred = pred / 255.0
        return pred

    if submitted:    
        u_img = Image.open(uploaded_file)
        
        # loading image    
        test_image = u_img
        image = load_prep_img(test_image)
        # img = keras.preprocessing.image.load_img(test_image, target_size=(IMAGE_SIZE))

        # Processing
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        # json input
        input_data_json_img = json.dumps({
            "signature_name": "serving_default",
            "instances": img_array.tolist(),
        })
        
        # predicting
        URL = 'https://img-classifier-backend.herokuapp.com/v1/models/image_classify_model:predict'
        response_img = requests.post(URL, data=input_data_json_img)
        response_img.raise_for_status() # raise an exception in case of error
        response_img = response_img.json()
        
        
        # predicting result
        score_img = tf.nn.softmax(response_img['predictions'][0])
        destination_by_img = class_names[np.argmax(score_img)]

        if destination_by_img == 'Sea':
            st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
            # st.("Here are the recommendations journey for you")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                image = Image.open('foto_rekomendasi/Sea/1.jpg')
                st.image(image, use_column_width=True, caption='Double Six Beach - 4.5⭐')
            with col4:
                image = Image.open('foto_rekomendasi/Sea/2.jpg')
                st.image(image, use_column_width=True, caption='Melasti Beach Ungasan - 4.8⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Sea/5.jpg')
                st.image(image, use_column_width=True, caption='Prama Sanur Beach Hotel Bali - 4.5⭐')
            col1, col2, col3, col4 = st.columns(4)
            with col3:
                image = Image.open('foto_rekomendasi/Sea/3.jpg')
                st.image(image, use_column_width=True, caption='Waterblow Park - 4.4⭐')
            with col2:
                image = Image.open('foto_rekomendasi/Sea/4.jpg')
                st.image(image, use_column_width=True, caption='Sanur Beach - 4.5⭐')
            st.markdown("---")
            
            # Mapping location
            st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
            filtered_data = datax[datax['target'] == destination_by_img] 
            # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
            
            st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude= -8.785502,
                        longitude=115.199806,
                        zoom = 10,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HexagonLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            radius=200,
                            elevation_scale=4,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            # if st.checkbox('Show Data'):
            #     st.subheader('Data')
            #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
            
            # Showing Description
            st.markdown("*Descriptions of each place on your destination*")
            st.write(data[data['target'] == destination_by_img][['name','description']].fillna("No description"))
            
        elif destination_by_img == 'Buildings':
            st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                image = Image.open('foto_rekomendasi/Buildings/1.jpg')
                st.image(image, use_column_width=True, caption='Beachwalk Shopping Center - 4.5⭐')
            with col4:
                image = Image.open('foto_rekomendasi/Buildings/2.jpg')
                st.image(image, use_column_width=True, caption='Uluwatu Hindu Temple - 4.6⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Buildings/5.jpg')
                st.image(image, use_column_width=True, caption='Discovery Mall Bali - 4.3⭐')
            col1, col2, col3, col4 = st.columns(4)
            with col2:
                image = Image.open('foto_rekomendasi/Buildings/3.jpg')
                st.image(image, use_column_width=True, caption='Ulun Danu Beratan Hindu Temple - 4.7⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Buildings/4.jpg')
                st.image(image, use_column_width=True, caption='Mal Bali Galeria - 4.5⭐')
                
            st.markdown("---")
            
            # Mapping location
            st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
            filtered_data = datax[datax['target'] == destination_by_img] 
            # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
            
            st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude= -8.785502,
                        longitude=115.199806,
                        zoom = 11,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HexagonLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            radius=200,
                            elevation_scale=4,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            # if st.checkbox('Show Data'):
            #     st.subheader('Data')
            #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
            
            # Showing Description
            st.markdown("*Descriptions of each place on your destination*")
            st.write(data[data['target'] == destination_by_img][['name','description']].fillna("No description"))
            
        elif destination_by_img == 'Forest':
            st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                image = Image.open('foto_rekomendasi/Forest/2.jpg')
                st.image(image, use_column_width=True, caption='Bali Zoo - 4.4⭐')
            with col4:
                image = Image.open('foto_rekomendasi/Forest/4.jpg')
                st.image(image, use_column_width=True, caption='Bali Swing - 4.4⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Forest/1.jpg')
                st.image(image, use_column_width=True, caption='Sacred Monkey Forest Sanctuary - 4.5⭐')
            col1, col2, col3, col4 = st.columns(4)
            with col2:
                image = Image.open('foto_rekomendasi/Forest/3.jpeg')
                st.image(image, use_column_width=True, caption='Kebun Raya Bali - 4.6⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Forest/5.jpg')
                st.image(image, use_column_width=True, caption='Bali Botanical Garden - 4.6⭐')
                
            st.markdown("---")
            
            # Mapping location
            st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
            filtered_data = datax[datax['target'] == destination_by_img] 
            # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
            
            st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude= -8.785502,
                        longitude=115.199806,
                        zoom = 10,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HexagonLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            radius=200,
                            elevation_scale=4,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            # if st.checkbox('Show Data'):
            #     st.subheader('Data')
            #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
            
            # Showing Description
            st.markdown("*Descriptions of each place on your destination*")
            st.write(data[data['target'] == destination_by_img][['name','description']].fillna("No description"))
        
        elif destination_by_img == 'Mountain':
            st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                image = Image.open('foto_rekomendasi/Mountain/3.jpg')
                st.image(image, use_column_width=True, caption='Bukit Catu Agro Bedugul - 4.3⭐')
            with col4:
                image = Image.open('foto_rekomendasi/Mountain/2.jpg')
                st.image(image, use_column_width=True, caption='Bukit Asah Desa Bugbug - 4.6⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Mountain/4.jpg')
                st.image(image, use_column_width=True, caption='Kajeng Rice Field - 4.5⭐')
            col1, col2, col3, col4 = st.columns(4)
            with col2:
                image = Image.open('foto_rekomendasi/Mountain/1.jpg')
                st.image(image, use_column_width=True, caption='Campuhan Ridge Walk - 4.5⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Mountain/5.jpg')
                st.image(image, use_column_width=True, caption='Gunung Lesung - 4.5⭐')
                        
            st.markdown("---")
            
            # Mapping location
            st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
            filtered_data = datax[datax['target'] == destination_by_img] 
            # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
            
            st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude= -8.459556,
                        longitude=115.046600,
                        zoom = 10,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HexagonLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            radius=200,
                            elevation_scale=4,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            # if st.checkbox('Show Data'):
            #     st.subheader('Data')
            #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
            
            # Showing Description
            st.markdown("*Descriptions of each place on your destination*")
            st.write(data[data['target'] == destination_by_img][['name','description']].fillna("No description"))
            
        else :
            st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                image = Image.open('foto_rekomendasi/Street/1.jpg')
                st.image(image, use_column_width=True, caption='Badung Market - 4.5⭐')
            with col4:
                image = Image.open('foto_rekomendasi/Street/2.jpg')
                st.image(image, use_column_width=True, caption='Sukawati Art Market - 4.3⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Street/4.jpg')
                st.image(image, use_column_width=True, caption='Pasar Senggol Gianyar - 4.4⭐')
            col1, col2, col3, col4 = st.columns(4)
            with col2:
                image = Image.open('foto_rekomendasi/Street/3.jpg')
                st.image(image, use_column_width=True, caption='Kreneng Market - 4.4⭐')
            with col3:
                image = Image.open('foto_rekomendasi/Street/5.jpg')
                st.image(image, use_column_width=True, caption='Pasar Batu Bulan Senggol - 4.3⭐')
                    
            st.markdown("---")
            
            # Mapping location
            st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
            filtered_data = datax[datax['target'] == destination_by_img] 
            # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
            
            st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude= -8.459556,
                        longitude=115.046600,
                        zoom = 10,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HexagonLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            radius=200,
                            elevation_scale=4,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            # if st.checkbox('Show Data'):
            #     st.subheader('Data')
            #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
            
            # Showing Description
            st.markdown("*Descriptions of each place on your destination*")
            st.write(data[data['target'] == destination_by_img][['name','description']].fillna("No description"))


    # ----------------------------------------------------------------------------------------------------- #
    # text input #

    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: black;'>or What Place Comes to Your Mind?🤔</h2>", unsafe_allow_html=True)
    # st.markdown("### ***Or what's in your mind?*** 🤔")
    text_input = st.text_area("Describe your destination place", placeholder = "Type here...")

    submitted_text = st.button('submit text')

    # if description1 == "":
    #     text_input = str("please select")
    # else:
    #     text_input = description1
    if submitted_text:
        target = 1 # default value // not impacting the model result

        isidata = [text_input, target]       
        columns = ['text', 'target']

        data_ = pd.DataFrame(data = [isidata], columns = columns)    

        # Menghilangkan kata-kata yang ada dalam list stopwords-english
        nltk.download('stopwords')

        # Fungsi untuk clean data
        def clean_text(text):
            '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
            and remove words containing numbers.'''
            text = str(text).lower() # Membuat text menjadi lower case
            text = re.sub('\[.*?\]', '', text) # Menghilangkan text dalam square brackets
            text = re.sub('https?://\S+|www\.\S+', '', text) # menghilangkan links
            text = re.sub('<.*?>+', '', text) # Menghilangkan text dalam <>
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # menghilangkan punctuatuion 
            text = re.sub('\n', '', text) # Menghilangkan enter / new line
            text = re.sub('\w*\d\w*', '', text) # Menghilangkan karakter yang terdiri dari huruf dan angka
            return text

        # cleaning data
        infdat = data_.drop('target', axis = 1)
        infdat['text'] = infdat['text'].apply(lambda x:clean_text(x))

        # Defining corpus with cleaned data
        ss = SnowballStemmer(language='english') 
        corpusinf = []
        for i in range(0, len(infdat)):
            decsr = infdat['text'][i]
            decsr = decsr.split()  # splitting data
            decsr = [ss.stem(word) for word in decsr if not word in stopwords.words('english')] # steeming setiap huruf dengan pengecualian kata yang ada dalam stopwords
            decsr = ' '.join(decsr)
            corpusinf.append(decsr)

        infdat['corpusinf'] = corpusinf
        infdat.reset_index(inplace = True)

        # encoding
        voc_size = 400
        inf_enc_corps = [one_hot(words, voc_size) for words in corpusinf]

        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Tokenization
        descr_length = 20
        inf_word_idx = tokenizer.texts_to_sequences(infdat['corpusinf'])
        inf_padded_seqs = pad_sequences(inf_word_idx, maxlen = descr_length)

        # json input
        input_data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": inf_padded_seqs.tolist(),
        })

        # Dictionary of each class
        classes_dict = { 0:'Buildings',
                        1:'Forest', 
                        2:'Mountain', 
                        3:'Sea', 
                        4:'Street' }

        # predicting
        URL = 'https://text-classifier-backend.herokuapp.com/v1/models/text_classification:predict'
        response = requests.post(URL, data=input_data_json)

        response = response.json()

        # predicting result
        score = tf.nn.softmax(response['predictions'][0])
        destination_by_text = classes_dict[np.argmax(score)]


        # ----------------------------------------------------------------------------------------------------- #
        # Recommending #

        if submitted:
            u_img = Image.open(uploaded_file)
            
            # loading image    
            test_image = u_img
            image = load_prep_img(test_image)
            # img = keras.preprocessing.image.load_img(test_image, target_size=(IMAGE_SIZE))

            # Processing
            img_array = keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)

            # json input
            input_data_json_img = json.dumps({
                "signature_name": "serving_default",
                "instances": img_array.tolist(),
            })
            
            # predicting
            URL = 'https://img-classifier-backend.herokuapp.com/v1/models/image_classify_model:predict'
            response_img = requests.post(URL, data=input_data_json_img)
            response_img.raise_for_status() # raise an exception in case of error
            response_img = response_img.json()
            
            
            # predicting result
            score_img = tf.nn.softmax(response_img['predictions'][0])
            destination_by_img = class_names[np.argmax(score_img)]

            if destination_by_img == 'Sea' or destination_by_text == 'Sea':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                # st.("Here are the recommendations journey for you")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Sea/1.jpg')
                    st.image(image, use_column_width=True, caption='Double Six Beach - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Sea/2.jpg')
                    st.image(image, use_column_width=True, caption='Melasti Beach Ungasan - 4.8⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Sea/5.jpg')
                    st.image(image, use_column_width=True, caption='Prama Sanur Beach Hotel Bali - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col3:
                    image = Image.open('foto_rekomendasi/Sea/3.jpg')
                    st.image(image, use_column_width=True, caption='Waterblow Park - 4.4⭐')
                with col2:
                    image = Image.open('foto_rekomendasi/Sea/4.jpg')
                    st.image(image, use_column_width=True, caption='Sanur Beach - 4.5⭐')
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
            elif destination_by_img == "Buildings" or destination_by_text == 'Buildings':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Buildings/1.jpg')
                    st.image(image, use_column_width=True, caption='Beachwalk Shopping Center - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Buildings/2.jpg')
                    st.image(image, use_column_width=True, caption='Uluwatu Hindu Temple - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Buildings/5.jpg')
                    st.image(image, use_column_width=True, caption='Discovery Mall Bali - 4.3⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Buildings/3.jpg')
                    st.image(image, use_column_width=True, caption='Ulun Danu Beratan Hindu Temple - 4.7⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Buildings/4.jpg')
                    st.image(image, use_column_width=True, caption='Mal Bali Galeria - 4.5⭐')
                    
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 11,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
            elif destination_by_img == "Forest" or destination_by_text == 'Forest':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Forest/2.jpg')
                    st.image(image, use_column_width=True, caption='Bali Zoo - 4.4⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Forest/4.jpg')
                    st.image(image, use_column_width=True, caption='Bali Swing - 4.4⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Forest/1.jpg')
                    st.image(image, use_column_width=True, caption='Sacred Monkey Forest Sanctuary - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Forest/3.jpeg')
                    st.image(image, use_column_width=True, caption='Kebun Raya Bali - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Forest/5.jpg')
                    st.image(image, use_column_width=True, caption='Bali Botanical Garden - 4.6⭐')
                    
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
            
            elif destination_by_img == "Mountain" or destination_by_text == 'Mountain':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Mountain/3.jpg')
                    st.image(image, use_column_width=True, caption='Bukit Catu Agro Bedugul - 4.3⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Mountain/2.jpg')
                    st.image(image, use_column_width=True, caption='Bukit Asah Desa Bugbug - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Mountain/4.jpg')
                    st.image(image, use_column_width=True, caption='Kajeng Rice Field - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Mountain/1.jpg')
                    st.image(image, use_column_width=True, caption='Campuhan Ridge Walk - 4.5⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Mountain/5.jpg')
                    st.image(image, use_column_width=True, caption='Gunung Lesung - 4.5⭐')
                            
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.459556,
                            longitude=115.046600,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
            else :
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Street/1.jpg')
                    st.image(image, use_column_width=True, caption='Badung Market - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Street/2.jpg')
                    st.image(image, use_column_width=True, caption='Sukawati Art Market - 4.3⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Street/4.jpg')
                    st.image(image, use_column_width=True, caption='Pasar Senggol Gianyar - 4.4⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Street/3.jpg')
                    st.image(image, use_column_width=True, caption='Kreneng Market - 4.4⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Street/5.jpg')
                    st.image(image, use_column_width=True, caption='Pasar Batu Bulan Senggol - 4.3⭐')
                        
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.459556,
                            longitude=115.046600,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                    
        else:
            if destination_by_text == 'Sea':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                # st.("Here are the recommendations journey for you")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Sea/1.jpg')
                    st.image(image, use_column_width=True, caption='Double Six Beach - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Sea/2.jpg')
                    st.image(image, use_column_width=True, caption='Melasti Beach Ungasan - 4.8⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Sea/5.jpg')
                    st.image(image, use_column_width=True, caption='Prama Sanur Beach Hotel Bali - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col3:
                    image = Image.open('foto_rekomendasi/Sea/3.jpg')
                    st.image(image, use_column_width=True, caption='Waterblow Park - 4.4⭐')
                with col2:
                    image = Image.open('foto_rekomendasi/Sea/4.jpg')
                    st.image(image, use_column_width=True, caption='Sanur Beach - 4.5⭐')
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
                
                    
            elif destination_by_text == 'Buildings':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Buildings/1.jpg')
                    st.image(image, use_column_width=True, caption='Beachwalk Shopping Center - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Buildings/2.jpg')
                    st.image(image, use_column_width=True, caption='Uluwatu Hindu Temple - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Buildings/5.jpg')
                    st.image(image, use_column_width=True, caption='Discovery Mall Bali - 4.3⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Buildings/3.jpg')
                    st.image(image, use_column_width=True, caption='Ulun Danu Beratan Hindu Temple - 4.7⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Buildings/4.jpg')
                    st.image(image, use_column_width=True, caption='Mal Bali Galeria - 4.5⭐')
                    
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 11,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
                    
                    
            elif destination_by_text == 'Forest':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Forest/2.jpg')
                    st.image(image, use_column_width=True, caption='Bali Zoo - 4.4⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Forest/4.jpg')
                    st.image(image, use_column_width=True, caption='Bali Swing - 4.4⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Forest/1.jpg')
                    st.image(image, use_column_width=True, caption='Sacred Monkey Forest Sanctuary - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Forest/3.jpeg')
                    st.image(image, use_column_width=True, caption='Kebun Raya Bali - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Forest/5.jpg')
                    st.image(image, use_column_width=True, caption='Bali Botanical Garden - 4.6⭐')
                    
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.785502,
                            longitude=115.199806,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
            
            elif destination_by_text == 'Mountain':
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Mountain/3.jpg')
                    st.image(image, use_column_width=True, caption='Bukit Catu Agro Bedugul - 4.3⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Mountain/2.jpg')
                    st.image(image, use_column_width=True, caption='Bukit Asah Desa Bugbug - 4.6⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Mountain/4.jpg')
                    st.image(image, use_column_width=True, caption='Kajeng Rice Field - 4.5⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Mountain/1.jpg')
                    st.image(image, use_column_width=True, caption='Campuhan Ridge Walk - 4.5⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Mountain/5.jpg')
                    st.image(image, use_column_width=True, caption='Gunung Lesung - 4.5⭐')
                            
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.459556,
                            longitude=115.046600,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))
                
            
            
            else :
                st.markdown("<h2 style='text-align: center; color: black;'>Start Your Journey Here</h2>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    image = Image.open('foto_rekomendasi/Street/1.jpg')
                    st.image(image, use_column_width=True, caption='Badung Market - 4.5⭐')
                with col4:
                    image = Image.open('foto_rekomendasi/Street/2.jpg')
                    st.image(image, use_column_width=True, caption='Sukawati Art Market - 4.3⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Street/4.jpg')
                    st.image(image, use_column_width=True, caption='Pasar Senggol Gianyar - 4.4⭐')
                col1, col2, col3, col4 = st.columns(4)
                with col2:
                    image = Image.open('foto_rekomendasi/Street/3.jpg')
                    st.image(image, use_column_width=True, caption='Kreneng Market - 4.4⭐')
                with col3:
                    image = Image.open('foto_rekomendasi/Street/5.jpg')
                    st.image(image, use_column_width=True, caption='Pasar Batu Bulan Senggol - 4.3⭐')
                        
                st.markdown("---")
                
                # Mapping location
                st.markdown("<h2 style='text-align: center; color: black;'>Your Journey Map 🗺️</h2>", unsafe_allow_html=True)
                filtered_data = datax[datax['target'] == destination_by_text] 
                # st.map(filtered_data) # Uncomment this jika ingin menggunakan st.map
                
                st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude= -8.459556,
                            longitude=115.046600,
                            zoom = 10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'HexagonLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=filtered_data,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                        ],
                    ))
                
                # if st.checkbox('Show Data'):
                #     st.subheader('Data')
                #     st.write(filtered_data) # uncomment jika ingin melihat filterred data
                
                # Showing Description
                st.markdown("*Descriptions of each place on your destination*")
                st.write(data[data['target'] == destination_by_text][['name','description']].fillna("No description"))

    st.markdown("---")
else:
    # banner
    image = Image.open('banner_page_2.png')
    st.image(image, use_column_width='auto', caption='Find your journey!')

    st.markdown("")
    st.markdown("")
    st.markdown("")

    # ----------------------------------------------------------------------------------------------------- #
    # image input input #
    st.markdown("<h2 style='text-align: center; color: black;'>What Traffic-Sign is it?🚸</h2>", unsafe_allow_html=True)
    # st.markdown("### ***Where do you want to go?*** 🏝️")
    uploaded_file = st.file_uploader("Upload the traffic-sign image")

    submitted = st.button('classify')

    # Dictionary of each class
    classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

    # assign image size
    IMAGE_SIZE = [30,30]

    def load_prep_img(filename):
        pred = np.array(filename)[:, :, :3]
        pred = tf.image.resize(pred, size = IMAGE_SIZE)
        pred = pred / 255.0
        return pred

    if submitted:    
        u_img = Image.open(uploaded_file)
        
        # loading image    
        test_image = u_img
        image = load_prep_img(test_image)
        # img = keras.preprocessing.image.load_img(test_image, target_size=(IMAGE_SIZE))

        # Processing
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        # json input
        input_data_json_img = json.dumps({
            "signature_name": "serving_default",
            "instances": img_array.tolist(),
        })
        
        # predicting
        URL = 'https://traffic-classifier-backend.herokuapp.com/v1/models/traffic_classifier:predict'
        response_img = requests.post(URL, data=input_data_json_img)
        response_img.raise_for_status() # raise an exception in case of error
        response_img = response_img.json()
        
        
        # predicting result
        score_img = tf.nn.softmax(response_img['predictions'][0])
        destination_by_img = classes[np.argmax(score_img)]
        
        # predicting result
        score = tf.nn.softmax(response_img['predictions'][0])
        st.markdown(
            "### This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(classes[np.argmax(score)], 100 * np.max(score))
        )