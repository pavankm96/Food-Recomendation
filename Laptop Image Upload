import streamlit as st
import tensorflow
import os
import numpy as np
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
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
import xgboost as xgb
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

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

db = init_with_service_account('food-recommendation-38053-firebase-adminsdk-s51o3-42bf7b6768.json')

start = [0]
passed = [0]
pack = [[]]
num = [0]

st.set_page_config(page_title="food recomendation", page_icon=":guardsman:", layout="wide")

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    #img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (299, 299, 3)
    img=Image.open(img_path)
    x = np.array(img.resize((299,299)))
    x=  x.reshape(1,299,299,3)
    #x=image.img_to_array(img_path)
    #st.text(x)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3) and return 4D tensor
    #return np.expand_dims(x, axis=0)
    return x
  

def img_analysis(img_path, plot=False): 
    # process image 
    img = path_to_tensor(img_path)
    img = preprocess_input(img)
    
    # make prediction 
    predicted_vec = model.predict(img)
    predicted_label = food101[np.argmax(predicted_vec)]
    
    # show predicted image 
    #img = cv2.imread(img_path)
    #rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    #st.image(img_path,width=1500)
    #plt.imshow(rgb)
    #plt.axis('off')
    #st.text("yummy! It looks like {}".format(predicted_label))
    pred_text="yummy! It looks like {}".format(predicted_label)
    plt.show()    
    return predicted_label,pred_text

def search_user_det(user_id):
    users_ref = db.collection(u'users')
    docs = users_ref.stream()
    abc={}
    for doc in docs:
        if doc.id==user_id:
            abc=doc.to_dict()
    return abc

def search_recent_login():
    users_ref = db.collection(u'users')
    docs = users_ref.stream()
    abc={}
    for doc in docs:
        if doc.id=='recent_login':
            abc=doc.to_dict()
            abc=abc["recent_login"]
    return abc

def predict(filename):
    result = []
    #pred=''
    # pack = []
    #print('total image', num[0])
    #for i in range(start[0], num[0]):
    pa = dict()

    filename = filename
    print('image filepath', filename)
    pred_img = filename
    pred,pred_text = img_analysis(pred_img)
    _true = pred
    pa['image'] = filename
    x = dict()
        
    #x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
    x[_true] = pred
    #x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
    #x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
    pa['result'] = x
    pa['nutrition'] = nutrition_table[_true]
    pa['food'] = nu_link+_true
    pa['idx'] = 'i - start[0]'
    pa['quantity'] = ''
    pa['recom'] = ''
    pack[0].append(pa)
    passed[0] += 1
    start[0] = passed[0]
    print('successfully packed')
    pred_value = pred
    #newly added code - start
    df=pd.read_csv('calorie_data.csv')
    e=list(df['categories'].values)
    print(pred_value)
    calo_per_wght=df[df['categories']== pred_value]['cal_per_weight'].values[0]
    print(calo_per_wght)
    #height=int(input("enter ur height"))
    #weight= int(input("enter ur weight")) 
    #gender= input("enter ur gender M for Male and F for Female")
    
    rec_log_id=search_recent_login()
    data=search_user_det(rec_log_id)
    height = int(data['height'])
    weight = int(data['weight'])
    gender = str(data['username']).strip()
    Diab=str(data['Diabetic']).strip()
    Chol=str(data['Cholestrol']).strip()
    bp=str(data['BP']).strip()
    alergy=str(data['Allergy']).strip()
    #Smoke=str(data['Smoking']).strip()
    #Alcohol=str(data['Alcohol']).strip()
    Age=int(str(data['Age']).strip())
    #st.text(str(height)+str(weight)+gender+Diab+Chol+bp+alergy+Smoke+Alcohol)
    #print(str(height)+str(weight)+gender+Diab+Chol+bp+alergy+Smoke+Alcohol)

    if gender=='M':
               gender_no=0
    else:
               gender_no=1
    #making datframe
    d = {'Height': [height], 'Weight': [weight],'Gender_no':[gender_no]}
    df1 = pd.DataFrame(data=d)
    #read the datset and split into test and train
    data=pd.read_csv('bmi_level.csv')
    labels=pd.DataFrame(data['Index'])
    features=data.drop(['Gender','Index'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.1,random_state=42)
    #clf=SVC(kernel='linear')
    clf=xgb.XGBClassifier()
    clf.fit(features,labels)
    print(df1)
    y_pred=clf.predict(df1)
    print("y_pred:",y_pred[0],"pred:",pred)
    #After prediction of obesity level we suggest the food to person
    recom_text = ''
    recom_text_dcb=''
    recom_obese=''
    yes_no=[]
    sugar_cat=['Apple pie','Baklava','Beignets','Bread pudding','Cannoli','Carrot cake','Cheesecake','Chocolate cake','Chocolate mousse','Churros','Creme brulee','Cupcakes','Donuts','Macarons','Pancakes','Panna cotta','Red velvet cake','Strawberry shortcake','Tiramisu','Waffles','french fries','pizza']
    if y_pred[0] ==4 or y_pred[0] ==5:
        if y_pred==4:
                recom_obese='You have Obesity'
        elif y_pred==5:
                recom_obese='You have Extream Obesity'
        if calo_per_wght >2.5 :
            recom_text="This food is of high calories"
            if(Diab=='Yes'):
                    yes_no.append('Diabeties')
            if (Chol=='Yes'):
                    yes_no.append('Cholestrol')
            if(bp=='Yes'):
                    yes_no.append("Blood Pressure")
            if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                recom_text="This food is of high calories so We don't recommended that"
                recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' have a long walk if you eat this!!!!! also you have'
                if(Age>=60):
                        recom_text="This food is of high calories so We don't recommended that"
                        recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' have a long walk if you eat this!!!!! also you are senior Citizen and you have'
                    
            else:
                    #st.text(" But you are very Healthy in Diet So you can eat this occationally ")
                    if recom_obese=='You have Obesity':
                        if(Age>=60):
                            recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' also you are senior Citizen'
                        else:
                            recom_text_dcb="Eat this occationally very little because "+recom_obese
                    elif recom_obese=='You have Extream Obesity':
                        if(Age>=60):
                            recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' also you are senior Citizen'
                        else:
                            recom_text_dcb="Just have a taste do not eat it because "+recom_obese
                    
        else :
            #st.text("You can eat that food It is of low calories")
            if pred in [sugar_food.lower() for sugar_food in sugar_cat]:
                if(Diab=='Yes'):
                    yes_no.append('Diabeties')
                if (Chol=='Yes'):
                    yes_no.append('Cholestrol')
                if(bp=='Yes'):
                    yes_no.append("Blood Pressure")
                if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                    recom_text="This food may have high in sugar"
                    recom_text_dcb='This food may have high in sugar do not eat it because you you have'
                    if(Age>=60):
                        recom_text="This food may have high in sugar"
                        recom_text_dcb='This food may have high in sugar do not eat it have a long walk if you eat this!!!!! also you are senior Citizen and you have'
            else:
                recom_text="You can eat that food It is of low calories"
    elif y_pred[0] ==3:
        recom_obese='You have Overweight'
        if calo_per_wght >3.5 :
            recom_text="This food is of high calories"
            if(Diab=='Yes'):
                yes_no.append('Diabeties')
            if (Chol=='Yes'):
                yes_no.append('Cholestrol')
            if(bp=='Yes'):
                yes_no.append("Blood Pressure")

            if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                recom_text="This food is of high calories so We don't recommended that"
                recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' have a long walk if you eat this!!!!! also you have'
                if(Age>=60):
                        recom_text="This food is of high calories so We don't recommended that"
                        recom_text_dcb='Just have a taste do not eat it because '+recom_obese+' have a long walk if you eat this!!!!! also you are senior Citizen and you have'

            else:
                    #st.text(" But you are very Healthy in Diet So you can eat this occationally ")
                    if(Age>=60):
                        recom_text_dcb='Eat this occationally very little because '+recom_obese+' also you are senior Citizen after eating take a good walk'
                    else:
                        recom_text_dcb="Eat this occationally very little because "+recom_obese
        else :
               #st.text("You can eat that food It is of low calories")
            if pred in [sugar_food.lower() for sugar_food in sugar_cat]:
                if(Diab=='Yes'):
                    yes_no.append('Diabeties')
                if (Chol=='Yes'):
                    yes_no.append('Cholestrol')
                if(bp=='Yes'):
                    yes_no.append("Blood Pressure")
                if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                    recom_text="This food may have high in sugar"
                    recom_text_dcb='This food may have high in sugar do not eat it because you you have'
                    if(Age>=60):
                        recom_text="This food may have high in sugar"
                        recom_text_dcb='This food may have high in sugar do not eat it have a long walk if you eat this!!!!! also you are senior Citizen and you have'
            else:
                recom_text="You can eat that food It is of low calories"

    elif y_pred[0] ==2:
        recom_obese='You have normal weight'
        if calo_per_wght >4.5 :
            recom_text="This food is of high calories"
            if(Diab=='Yes'):
                    yes_no.append('Diabeties')
            if (Chol=='Yes'):
                    yes_no.append('Cholestrol')
            if(bp=='Yes'):
                    yes_no.append("Blood Pressure")

            if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                recom_text="This food is of high calories so We don't recommended that"
                recom_text_dcb='Eat this occationally very little because '+recom_obese+' have a long walk if you eat this!!!!! also you have'
                if(Age>=60):
                        recom_text="This food is of high calories so We don't recommended that"
                        recom_text_dcb='Eat this occationally very little because '+recom_obese+' have a long walk if you eat this!!!!! also you are senior Citizen and you have'

            else:
                    recom_text="You can eat this food because you"+recom_obese+"but still eat ocationally"
                    #st.text(" But you are very Healthy in Diet So you can eat this occationally ")
                    if(Age>=60):
                        recom_text_dcb='Eat this occationally very little because '+recom_obese+' also you are senior Citizen after eating take a good walk'
                    else:
                        recom_text_dcb="But you are very Healthy in Diet So you can eat this occationally because "+recom_obese
        else :
               #st.text("You can eat that food It is of low calories")
            if pred in [sugar_food.lower() for sugar_food in sugar_cat]:
                if(Diab=='Yes'):
                    yes_no.append('Diabeties')
                if (Chol=='Yes'):
                    yes_no.append('Cholestrol')
                if(bp=='Yes'):
                    yes_no.append("Blood Pressure")
                if(Diab=='Yes' or Chol=='Yes' or bp=='Yes'):
                    recom_text="This food may have high in sugar"
                    recom_text_dcb='This food may have high in sugar do not eat it because you you have'
                    if(Age>=60):
                        recom_text="This food may have high in sugar"
                        recom_text_dcb='This food may have high in sugar do not eat it have a long walk if you eat this!!!!! also you are senior Citizen and you have'
            
            recom_text="You can eat that food It is of low calories"

    elif y_pred[0] ==1:
        #st.text("You can eat that food")
        recom_text="You can eat that food"
    
    


    #newly added code - end


    pack[0][-1]['recom'] = recom_text
    
    print(pack)
    # compute the average source of calories
    for p in pack[0]:
        nutrients[0]['value'] = (nutrients[0]['value'] + p['nutrition'][0]['value'])
        nutrients[1]['value'] = (nutrients[1]['value'] + p['nutrition'][1]['value'])
        nutrients[2]['value'] = (nutrients[2]['value'] + p['nutrition'][2]['value'])
        nutrients[3]['value'] = (nutrients[3]['value'] + p['nutrition'][3]['value'])
        nutrients[4]['value'] = (nutrients[4]['value'] + p['nutrition'][4]['value'])

    
    #nutrients[0]['value'] = nutrients[0]['value'] / num[0]
    #nutrients[1]['value'] = nutrients[1]['value'] / num[0]
    #nutrients[2]['value'] = nutrients[2]['value'] / num[0]
    #nutrients[3]['value'] = nutrients[3]['value'] / num[0]
    #nutrients[4]['value'] = nutrients[4]['value'] / num[0]



    #return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)
    return recom_text,recom_text_dcb,pred_text,yes_no,pa['food']


nu_link = 'https://www.nutritionix.com/food/'

# Loading the best saved model to make predictions.
#tensorflow.keras.backend.clear_session()
#model_best = load_model('best_model_101class.hdf5', compile=False)
#print('model successfully loaded!')

n_classes = 101
weght_path="inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

#upload_file=st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png", "gif"])
#file_path = st.file_uploader.get_file_path_or_buffer(uploaded_file)
import streamlit as st
from PIL import Image
import numpy as np

upload_file=st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png", "gif"])


weght_path="inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
# base model is inception_v3 weights pre-trained on ImageNet
base_model = InceptionV3(
    weights=None, 
    include_top=False,
    input_shape=(299,299,3)
)
base_model.load_weights(weght_path)

x = base_model.output 

# added layers to the base model
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.4)(x)
x = Flatten()(x)

predictions = Dense(n_classes, activation='softmax')(x)    
model_file = 'food101_final_model.h5'
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(model_file)
## Load the class labels (which are indexes are the same as the ones from generator)
with open('labels.txt', 'r') as f: 
    food101 = [l.strip().lower() for l in f]

if upload_file is not None:
    recom_text,recom_text_dcb,pred_text,yes_no,more_values=predict(upload_file)

    col1,col2=st.columns([10,10])
    st.markdown("<br>",unsafe_allow_html=True)
    col3,col4=st.columns([10,10])
    html="<p> <b><h1>"+pred_text+"</h1><br><h2>"+recom_text+"</h2><br><h1>"+recom_text_dcb+"</h1></b></p>"
    if yes_no is not None:
        if len(yes_no)==1:
            for i in yes_no:
                recom_text_dcb=recom_text_dcb+" "+i
            html="<p> <b><h1>"+pred_text+"<br></h1><h2>"+recom_text+"</h2><br><h1>"+recom_text_dcb+"</h1></b></p>"
        else:
            for i in yes_no:
                recom_text_dcb=recom_text_dcb+"<br>"+i
            html="<p> <b><h1>"+pred_text+"</h1><br><h2>"+recom_text+"</h2><br><h1>"+recom_text_dcb+"</h1></b></p>"
    with col1:
        st.image(upload_file,width=600)
    with col2:
        st.markdown(html,unsafe_allow_html=True)
    with col3:
        nutrients=pd.DataFrame(nutrients)
        labels = 'Protien', 'Calcium', 'Fat', 'Carbohydrates', 'Vitamins'
        sizes = list(nutrients['value'])
        max_size_pos=max(sizes)
        max_size_pos=sizes.index(max_size_pos)
        explode = (0,0,0,0,0)  
        explode=list(explode)
        explode[max_size_pos]=0.1
        explode=tuple(explode)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
        ax1.axis('equal') 
        plt.show()
        st.markdown("<h1> <center> Nutrtional Graph </center> </h1><br>",unsafe_allow_html=True)
        st.pyplot(fig1)

    with col4:
        nutrients=pd.DataFrame(nutrients)
        nutrients=nutrients.style.set_table_styles(
            [{'selector': 'th.col_heading',
              'props': [('background-color', 'maroon'),("font-size", "55px"),("border", "3px solid black"),("text-align","center"),("color","white")]},
             {'selector': 'td',
              'props': [("font-size", "50px"),("border", "3px solid black"),("text-align","center"),("background-color","yellow")]},
             {'selector': 'tr',
              'props': [('background-color', '#f4f4f4')]}])
        nutrients=nutrients.hide_index()
        st.markdown("<h1> <center> Nutrtional Values </center> </h1><br>",unsafe_allow_html=True)
        st.markdown("<center>"+nutrients.to_html()+"</center>",unsafe_allow_html=True)

    more_nutri="<center><h1><a href="+more_values+"> Click Hear for more Nutritional Values</a></h1></center>"
    st.markdown(more_nutri,unsafe_allow_html=True)

print("nutri=",nutrients)
