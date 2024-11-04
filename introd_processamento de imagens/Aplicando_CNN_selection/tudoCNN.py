import os
import gc
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
import tensorflow as tf


# 001
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, InceptionV3, EfficientNetB0, Xception, MobileNetV2
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras import backend as K


# :================================================================================================================

# Criando função para calcular as métricas

def get_metrics(y_true, y_pred):
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (vp + vn) / (vp + fp + fn + vn)
    recall = vp / (vp + fn)
    specificity = vn / (vn + fp)
    precision = vp / (vp + fp)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'recall': recall,
        'precision': precision,
        'f1-score': f1,
        'kappa': kappa,
        'auc-roc': auc_roc
    }


# def get_metrics(y_true, y_pred):
    # # Confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    
    # # Accuracy
    # accuracy = np.trace(cm) / float(np.sum(cm))
    
    # # Recall, Specificity, Precision
    # recall = np.diag(cm) / np.sum(cm, axis=1)
    # specificity = np.diag(cm) / np.sum(cm, axis=0)
    # precision = np.diag(cm) / np.sum(cm, axis=0)
    
    # # Outros
    # f1 = f1_score(y_true, y_pred)
    # kappa = cohen_kappa_score(y_true, y_pred)
    # auc_roc = roc_auc_score(y_true, y_pred)
    
    # return {
    #     'accuracy': accuracy,
    #     'recall': np.mean(recall),
    #     'specificity': np.mean(specificity),
    #     'precision': np.mean(precision),
    #     'f1-score': f1,
    #     'kappa': kappa,
    #     'auc-roc': auc_roc
    # }


# :================================================================================================================

# Criando função para seleção de esquema de cor 
def convert_color_scale(image, scale):
    if scale == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif scale == 'rgb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif scale == 'grayscale':
        # Converter para escala de cinza e replicar para 3 canais
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.merge([gray, gray, gray])
    elif scale == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif scale == 'luv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif scale == 'xyz':
        return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    else:
        raise ValueError("Escala de cor não suportada.")
    

# :================================================================================================================

    # Carregamento e pré-processamento de imagens com escolha de escala de cor
# def load_images(folder, color_scale,img_extensions):
#     images = []
#     for filename in os.listdir(folder):
#         if any(filename.lower().endswith(ext) for ext in img_extensions):
#             img_path = os.path.join(folder, filename)
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (224, 224))  # Ajuste o tamanho conforme necessário

#             # Converta para a escala de cor desejada
#             img = convert_color_scale(img, color_scale)

#             # Se a imagem estiver em escala de cinza, expanda as dimensões
#             if color_scale == 'grayscale':
#                 img = np.expand_dims(img, axis=-1)  # Adiciona uma dimensão de canal

#             images.append(img)
#     return np.array(images) 
def load_images(folder, color_scale, img_extensions):
    images = []
    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in img_extensions):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Ajuste o tamanho conforme necessário
                img = convert_color_scale(img, color_scale)  # Converta para a escala de cor desejada
                # Se a imagem estiver em escala de cinza, expanda as dimensões
                if color_scale == 'grayscale':
                    img = np.expand_dims(img, axis=-1)  # Adiciona uma dimensão de canal
                images.append(img)
            else:
                print(f"Erro ao carregar a imagem: {img_path}")
    return np.array(images)

# :================================================================================================================

# Defina as pastas de dados
data_dir = r"C:\Users\andre\Downloads\archive\OvarianCancer\\duas"
normal_dir = os.path.join(data_dir, 'Non_Cancerous')
cancer_dir = os.path.join(data_dir, 'Endometri')
img_extensions = ['.jpg', '.jpeg', '.png']


# :================================================================================================================

# Criar um DataFrame para armazenar os resultados
columns = ['Modelo', 'Acuracia', 'Sensibilidade', 'Especificidade', 'F-Score', 'AUC-ROC']
df_metrics = pd.DataFrame(columns=columns)

# Definir o caminho para salvar o melhor modelo
model_checkpoint_path = "best_model.h5"

# :================================================================================================================

# Iterar sobre os modelos
def model_execution(model_name):

    # Carregamento de imagens e conversão para XYZ
    normal_images = load_images(normal_dir, 'xyz',img_extensions)
    cancer_images = load_images(cancer_dir, 'xyz',img_extensions)

    # Rótulos para imagens (0 para normal, 1 para câncer)
    normal_labels = np.zeros(normal_images.shape[0])
    cancer_labels = np.ones(cancer_images.shape[0])

    # Concatenar imagens e rótulos
    all_images = np.concatenate([normal_images, cancer_images], axis=0)
    all_labels = np.concatenate([normal_labels, cancer_labels], axis=0)

    # Dividir o conjunto de dados em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # Escolha entre 5 tipos de conversão de escala de cor aqui (por exemplo, normalização)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Escolher o modelo de CNN
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'VisionTransformer':
            base_model = VitB16(image_size=(224, 224), weights='imagenet', include_top=False)
    else:
        raise ValueError("Modelo de CNN não suportado.")

    # Construir o modelo
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compilar o modelo
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Configurar o retorno de chamada ModelCheckpoint
#     checkpoint = callbacks.ModelCheckpoint(model_checkpoint_path,
#                                            monitor='val_accuracy',  # Métrica a ser monitorad
#                                            mode='max',               # Salvar o modelo com a maior precisão
#                                            verbose=1)

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=1, batch_size=16, validation_split=0.2)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calcular métricas
    metrics = get_metrics(y_test, y_pred_binary)

    # Adicionar as métricas ao DataFrame
    metrics['Modelo'] = model_name
    
    # Excluir o modelo atual para liberar memória da GPU   
    del all_images, normal_images, cancer_images, X_train, X_test, y_train, y_test, y_pred, y_pred_binary
    del all_labels, normal_labels, cancer_labels
    tf.keras.backend.clear_session()
    del model, base_model
    gc.collect()
    
    return metrics


# :================================================================================================================

chosen_model = 'VGG16'
metrics = model_execution(chosen_model)
temp_df = pd.DataFrame([[chosen_model, metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1-score'], metrics['auc-roc']]], columns=columns)  
df_metrics = pd.concat([df_metrics, temp_df], ignore_index=True)

chosen_model = 'ResNet50'
metrics = model_execution(chosen_model)
temp_df = pd.DataFrame([[chosen_model, metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1-score'], metrics['auc-roc']]], columns=columns)  
df_metrics = pd.concat([df_metrics, temp_df], ignore_index=True)

chosen_model = 'InceptionV3'
metrics = model_execution(chosen_model)
temp_df = pd.DataFrame([[chosen_model, metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1-score'], metrics['auc-roc']]], columns=columns)  
df_metrics = pd.concat([df_metrics, temp_df], ignore_index=True)

chosen_model = 'MobileNetV2'
metrics = model_execution(chosen_model)
temp_df = pd.DataFrame([[chosen_model, metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1-score'], metrics['auc-roc']]], columns=columns)  
df_metrics = pd.concat([df_metrics, temp_df], ignore_index=True)

chosen_model = 'EfficientNetB0'
metrics = model_execution(chosen_model)
temp_df = pd.DataFrame([[chosen_model, metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1-score'], metrics['auc-roc']]], columns=columns)  
df_metrics = pd.concat([df_metrics, temp_df], ignore_index=True)

print(df_metrics)

