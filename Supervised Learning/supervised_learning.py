path = " "


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
from dython.nominal import associations

#%% Chargement de données
person = pd.read_csv(path+'learn_dataset.csv')
job = pd.read_csv(path+'learn_dataset_job.csv')
city_admin = pd.read_csv(path+'city_adm.csv')
city_loc = pd.read_csv(path+'city_loc.csv')
person_test = pd.read_csv(path+'test_dataset.csv')
department = pd.read_csv(path+'departments.csv')
              
#%% Target Distribution
plt.box(False)
plt.grid('whitegrid', color = 'silver', alpha = 0.3)
sns.countplot(x = person['target'], palette = 'GnBu') 

#%% Regroupement des commune
person_admin = pd.merge(person, city_admin, how = 'left', on = 'INSEE_CODE')
person_loc = pd.merge(person_admin, department, how = 'left', on = 'Dep')

#%% Correlation
corr_3 = associations(person_loc.drop(columns = ['Primary_key',
                                                 'Nom de la commune',
                                                 'INSEE_CODE',
                                                 'Town_type',         
                                                 'Dep',
                                                 'REG']))['corr']
plt.figure(figsize = (8,6))
sns.heatmap(corr_3, annot = True, cmap="GnBu")

#%% Division des données - tous les variables
X_train, X_test, y_train, y_test = train_test_split(person_loc.drop(columns = ['Primary_key',
                                                                               'Nom de la commune', 
                                                                               'target', 
                                                                               'INSEE_CODE', 
                                                                               'Town_type', 
                                                                               'Dep', 
                                                                               'REG']),
                                                    person['target'], test_size = 0.2, random_state = 1)


#%% Division des données
X_train, X_test, y_train, y_test = train_test_split(person_loc.drop(columns = ['Primary_key',
                                                                               'job_42', 
                                                                               'CURRENT_AGE', 
                                                                               'is_student', 
                                                                               'household_type', 
                                                                               'Nom de la commune', 
                                                                               'target', 
                                                                               'INSEE_CODE', 
                                                                               'Town_type', 
                                                                               'Dep', 
                                                                               'REG']),
                                                    person['target'], test_size = 0.2, random_state = 1)

#%% Dichotomisation des données
X_train_encodded = pd.get_dummies(X_train, drop_first = True )
X_test_encodded = pd.get_dummies(X_test, drop_first = True )

#%% Logistic Regression GridSearch
lr_1 = LogisticRegression(class_weight = 'balanced', random_state = 0)
parametres = {'solver': ['liblinear', 'saga'], 'C': [0.1, 1, 10]} 
grid_lr_1 = GridSearchCV(lr_1, param_grid = parametres, verbose = 2, cv = StratifiedKFold(n_splits=3, random_state = 0, shuffle = True))
grid_lr_1 = grid_lr_1.fit(X_train_encodded, y_train)
print(grid_lr_1.best_params_)

y_pred_lr_1 = grid_lr_1.predict(X_test_encodded)
pd.crosstab(y_pred_lr_1, y_test, colnames = ['Real Class'], rownames = ['Predicted Class'])

print(classification_report(y_test, y_pred_lr_1))

grid_lr_1.score(X_test_encodded, y_test)

#%% Random Forest GridSearch
rf_1 = RandomForestClassifier(class_weight = 'balanced', oob_score = True, random_state = 0)
parametres = {'n_estimators': [50, 100]}
grid_rf_1 = GridSearchCV(rf_1, param_grid = parametres, verbose = 2, cv = StratifiedKFold(n_splits=3, random_state = 0, shuffle = True))
grid_rf_1 = grid_rf_1.fit(X_train_encodded, y_train)
print(grid_rf_1.best_params_)

y_pred_rf_1 = grid_rf_1.predict(X_test_encodded)
print(pd.crosstab(y_pred_rf_1, y_test, colnames = ['Real Class'], rownames = ['Predicted Class']))

print(classification_report(y_test, y_pred_rf_1))

grid_rf_1.score(X_test_encodded, y_test)

#%% SVM - GridSearch
csvm_1 = SVC(class_weight = 'balanced', probability = True, random_state = 0)
parametres = {'C':[0.1,1], 'kernel':['rbf','poly']}       
grid_svm_1 = GridSearchCV(csvm_1, param_grid = parametres, verbose = 2, cv = StratifiedKFold(n_splits=3, random_state = 0, shuffle = True))
grid_svm_1 = grid_svm_1.fit(X_train_encodded, y_train)
grid_svm_1.best_params_

y_pred_grid_svm_1 = grid_svm_1.predict(X_test_encodded)
pd.crosstab(y_pred_grid_svm_1, y_test, colnames = ['Real Class'], rownames = ['Predicted Class'])

print(classification_report(y_test, y_pred_grid_svm_1))

grid_svm_1.score(X_test_encodded, y_test)

#%% celui avec le plus haut taux de bonne précision moyen et le plus faible écart-type.

pd.set_option('display.max_columns', None)

lr_result_1 = pd.DataFrame(grid_lr_1.cv_results_)[['params','mean_test_score','std_test_score']]
rf_result_1 = pd.DataFrame(grid_rf_1.cv_results_)[['params','mean_test_score','std_test_score']]
svm_result_1 = pd.DataFrame(grid_svm_1.cv_results_)[['params','mean_test_score','std_test_score']]

best_lr_1 = lr_result_1[lr_result_1['params'] == grid_lr_1.best_params_]
best_rf_1 = rf_result_1[rf_result_1['params'] == grid_rf_1.best_params_]
best_svm_1 = svm_result_1[svm_result_1['params'] == grid_svm_1.best_params_]

best_models_1 = pd.concat([best_lr_1, best_rf_1, best_svm_1])
model_names = ['Logistic Regression', 'Random Forest', 'SVM']

best_models_1.insert(0, 'Model', model_names)
best_models_1

#%% ROC - à comparer entre différents modèles sur même graphe
probs_lr_1 = grid_lr_1.predict_proba(X_test_encodded)
fp_lr_1, tp_lr_1, threshold_lr_1 = roc_curve(y_test, probs_lr_1[:,1], pos_label = 'Y')
roc_auc_lr_1 = auc(fp_lr_1, tp_lr_1)

probs_rf_1 = grid_rf_1.predict_proba(X_test_encodded)
fp_rf_1, tp_rf_1, threshold_rf_1 = roc_curve(y_test, probs_rf_1[:,1], pos_label = 'Y')
roc_auc_rf_1 = auc(fp_rf_1, tp_rf_1)

probs_svm_1 = grid_svm_1.predict_proba(X_test_encodded)
fp_svm_1, tp_svm_1, threshold_svm_1 = roc_curve(y_test, probs_svm_1[:,1], pos_label = 'Y')
roc_auc_svm_1 = auc(fp_svm_1, tp_svm_1)

#%% ROC
plt.figure(figsize=(8,6))
plt.box(False)
plt.grid(axis = 'y', color = 'silver', linewidth = 0.5)
plt.tick_params(bottom = False, left = False)
plt.plot(fp_lr_1, tp_lr_1, color='orange', label ='Logistic Regression : auc = %0.2f' % roc_auc_lr_1)
plt.plot(fp_rf_1, tp_rf_1, color='red', label = 'Random Forest : auc = %0.2f' % roc_auc_rf_1)
plt.plot(fp_svm_1, tp_svm_1, color='green', label = 'SVM : auc = %0.2f' % roc_auc_svm_1)
plt.plot((0,1),(0,1), 'b--', label='Random : auc = 0.5')
plt.title('Courbe ROC', size=12)
plt.ylabel('Taux vrais positifs')
plt.xlabel('Taux faux positifs') 
plt.legend(loc='lower right', frameon=False);

#%% A présent, il nous reste à sélectionner et entraîner le meilleur algorithme, 

threshold_1 = pd.DataFrame({'threshold': threshold_lr_1,'tp-fp' : tp_lr_1 - fp_lr_1})
threshold_1[threshold_1['tp-fp'] == max(threshold_1['tp-fp'])]

#%% Meilleur Modèle
threshold = 0.497212
y_pred_prob_lr_1 = np.where(probs_lr_1[:,1] > threshold, 'Y', 'J')

print(pd.crosstab(y_pred_prob_lr_1, y_test, 
                  colnames = ['Real Class'], 
                  rownames = ['Predicted Class']))

print(classification_report(y_test, y_pred_prob_lr_1))


#%% Enregistrement des modèles
pickle.dump(grid_lr_1, open('grid_lr_1.pkl', 'wb'))
pickle.dump(grid_rf_1, open('grid_rf_1.pkl', 'wb'))
pickle.dump(grid_svm_1, open('grid_svm_1.pkl', 'wb'))
pickle.dump(best_models_1, open('best_models_1.pkl', 'wb'))

#%% Predictions
test_admin = pd.merge(person_test, city_admin, how = 'left', on = 'INSEE_CODE')
test_loc = pd.merge(test_admin, department, how = 'left', on = 'Dep')

test = test_loc.drop(columns = ['Primary_key',
                                'Nom de la commune', 
                                'INSEE_CODE', 
                                'Town_type', 
                                'Dep', 
                                'REG'])

test_encodded = pd.get_dummies(test, drop_first = True )

predictions_prob_1 = grid_lr_1.predict_proba(test_encodded)
predictions_1 = np.where(predictions_prob_1[:,1] > threshold, 'Y', 'J')

#%% Saving
predictions_v1 = pd.DataFrame({'Primary_key': person_test['Primary_key'], 'target': predictions_1})
predictions_v1.to_csv(path+'predictions_v1.csv', sep =',', index = False)

