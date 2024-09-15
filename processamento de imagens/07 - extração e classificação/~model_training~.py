from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost
import pandas
import pickle

def get_metrics(y_true, y_pred):
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (vp + vn) / (vp + fp + fn + vn)
    recall = vp / (vp + fn)
    specificity = vn / (vn + fp)
    precision = vp / (vp + fp)
    
    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'recall': recall,
        'precision': precision,
    }


df = pandas.read_csv('C:\\Users\\andre\\OneDrive\\Documentos\\atividades\\estudos_opencv\\processamento de imagens\\07 - extração e classificação\\dados.csv', delimiter = ';')

df['Label'] = df['Label'].astype(int)   # convertendo os dados da coluna para inteiros

X = df.drop('Label', axis = 1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

balanced = SMOTE(random_state=42)
X_train, y_train = balanced.fit_resample(X_train, y_train)

model = xgboost.XGBClassifier(objective = "binary:logistic", random_state = 42, max_depth = 9, colsample_bytree = 0.4033,
                                        min_child_weight = 6, gamma = 0.429, eta = 0.5995, n_estimators = 1000,
                                        use_label_encoder=False, eval_metric='merror')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = get_metrics(y_test, y_pred)

print(metrics)

# pickle.dump(model, open('C:/Users/andre/Documents/atividade/teste/opencv/07 - extração e classificação', 'wb'))

pickle.dump(model, open('C:\\Users\\andre\\OneDrive\\Documentos\\atividades\\estudos_opencv\\processamento de imagens\\07 - extração e classificação\\modelo.pkl', 'wb'))
