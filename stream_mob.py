import streamlit as st
import tensorflow
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
import numpy as np 
from PIL import Image
import subprocess
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin

st.set_page_config(page_title="User Login and Signup", page_icon=":guardsman:", layout="wide")
# Use a service account.
#firebase_admin.delete_app(firebase_admin.get_app())

#cred = credentials.Certificate('D:/Food Detector Streamlit/food-recommendation-38053-firebase-adminsdk-s51o3-42bf7b6768.json')
#app = firebase_admin.initialize_app(cred)

def init_with_service_account(file_path):
     """
     Initialize the Firestore DB client using a service account
     :param file_path: path to service account
     :return: firestore
     """
     cred = credentials.Certificate(file_path)
     try:
         firebase_admin.get_app()
     except ValueError:
         firebase_admin.initialize_app(cred)
     return firestore.client()

db=init_with_service_account('D:/Food Detector Streamlit/food-recommendation-38053-firebase-adminsdk-s51o3-42bf7b6768.json')
#db = firestore.client()

def login_user(user_id, password):
    collection_ref=db.collection('users')
    query = collection_ref.where('user_id', '==', user_id).where('password', '==', password)
    docs=query.stream()

    if len(list(docs)) == 1:
        return True
    else:
        return False
def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(200, 200))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img = img / 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    foods_sorted.sort()
    pred_value = foods_sorted[index]
    if show:
        st.image(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()

def main():
    st.markdown("<style>input {font-size: 40px;}</style>", unsafe_allow_html=True)

    menu = ["Homepage", "Login", "Signup"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Homepage":
        st.write("Welcome to the Homepage")

    elif choice == "Login":
        st.write("Login")
        user_id=st.text_input("User Id")
        password = st.text_input("Password", type="password")

        if st.button("Submit"):
            if login_user(user_id, password):
                st.success("Login Successful")
                rec_log=open('recent_login.txt','w')
                rec_log.write(str(user_id))
                rec_log.close()
                subprocess.run(["streamlit", "run", "food_det_mob.py"])
                    
            else:
                st.error("Login Unsuccessful!!!! Please check User Id and Password")

    elif choice == "Signup":
        menu_yes_no=['Not Selected','No','Yes']
        menu_gender=['Not Selected','M','F']
        st.write("Signup")
        username = st.text_input("Name")
        user_id=st.text_input("User Id")
        password = st.text_input("Password", type="password")
        Gender=st.selectbox("Gender",menu_gender)
        Age=st.text_input("Age")
        Diabetic=st.selectbox("Diabetic",menu_yes_no)
        Cholestrol=st.selectbox("Cholestrol",menu_yes_no)
        BP=st.selectbox("BP",menu_yes_no)
        Allergy=st.selectbox("Allergy",menu_yes_no)
        weight=st.text_input("weight")
        height=st.text_input("height")
        Smoking=st.selectbox("Smoking",menu_yes_no)
        Alcohol=st.selectbox("Alcohol",menu_yes_no)
        if st.button("Submit"):
            doc_ref = db.collection(u'users').document(u'{}'.format(user_id))
            doc_ref.set({
        u'username': u'{}'.format(username),
        u'user_id':u'{}'.format(user_id),
        u'password': u'{}'.format(password),
        u'Gender': u'{}'.format(Gender),
        u'Age': u'{}'.format(Age),
        u'Diabetic': u'{}'.format(Diabetic),
        u'Cholestrol': u'{}'.format(Cholestrol),
        u'BP': u'{}'.format(BP),
        u'Allergy': u'{}'.format(Allergy),
        u'weight': u'{}'.format(weight),
        u'height': u'{}'.format(height),
        u'Smoking': u'{}'.format(Smoking),
        u'Alcohol': u'{}'.format(Alcohol)
        })

    #db.close()
    #firebase_admin.delete_app(firebase_admin.get_app())

if __name__ == '__main__':
    main()
