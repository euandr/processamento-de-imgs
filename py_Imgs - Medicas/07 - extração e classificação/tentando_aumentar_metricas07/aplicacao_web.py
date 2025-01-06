import streamlit.components.v1 as components
import lime.lime_tabular
import streamlit
import mahotas
import pickle
import pandas
import numpy
import cv2

from sklearn.model_selection import train_test_split

def get_model():
    return pickle.load(open(r'C:\Users\andre\Documents\atividade\estudos_opencv\processamento de imagens\07 - extração e classificação\\modelo.pkl', 'rb'))

def convert_byteio_image(string):
    array = numpy.frombuffer(string, numpy.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

streamlit.markdown("<h1 style='text-align: center; color: yellow;'>Aplicação web para analize de imagens com alguns tipos de  glóbulos brancos no sangue</h1>", unsafe_allow_html=True)

streamlit.sidebar.title('configurações')

uploaded_image = streamlit.sidebar.file_uploader("Escolha uma imagem", type = 'jpg', accept_multiple_files = False)

model = get_model()

if uploaded_image is not None:

    bytes_data = uploaded_image.getvalue()

    image = convert_byteio_image(bytes_data)

    if (image.shape != (256, 256)):    
        image = cv2.resize(image, (256, 256))

    features = mahotas.features.haralick(image, compute_14th_feature = True, return_mean = True).reshape(1, 14)

    pred = model.predict(features)   

    probs = model.predict_proba(features)  

    streamlit.markdown("<h3 style='text-align: center; color: red;'>Image</h3>", unsafe_allow_html=True)

    col1, col2, col3 = streamlit.columns([0.2, 5, 0.2])

    col2.image(image, use_column_width=True)

    pred_output = "paciente contem globulos brancos EOSINOPHIL, com {:.2%} de certeza".format(probs[0][1]) if pred[0] == 1 else "paciente contem globulos brancos NEUTROPHIL, com {:.2%} de certeza".format(probs[0][0]) 

    streamlit.markdown("<h4 style='text-align: center; color: red;'>" + pred_output + "</h4>", unsafe_allow_html=True)

if streamlit.sidebar.button("Explique a previsão"):

    streamlit.markdown("<h3 style='text-align: center; color: red;'>Interpretacao</h3>", unsafe_allow_html=True)

    with streamlit.spinner('calculando...'):        

        df = pandas.read_csv(r'C:\Users\andre\Documents\atividade\estudos_opencv\processamento de imagens\07 - extração e classificação\\dados2.csv', delimiter = ';')

        X = df.drop('Label', axis = 1)
        y = df['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = X.columns, class_names = ['EOSINOPHIL', 'NEUTROPHIL'], 
                                                                feature_selection = 'lasso_path', discretize_continuous = True)
        
        exp = explainer.explain_instance(features.reshape(14,), model.predict_proba, num_features = 14)
        
        components.html(exp.as_html(predict_proba = False), height = 800)
