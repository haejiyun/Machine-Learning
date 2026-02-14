import streamlit as st
import os
from PIL import Image


st.header("Classification bimodale avec un modèle DL")

st.write("Commentaires sur quelques dimensionnements de ce modèle :")
st.write("1- longueur du texte à 200 mots qui embarque 98 pour cent des textes sans les tronquer")
st.write("2- la définition choisie des images de 150 sur 150 pixels pour le modéle VGG16 car la médiane du poids des images est de 50 ko")
st.write("3- de nombreux essais empiriques sur la longueur de vecteur dense de sortie de la couche embedding")
st.write("4- itérations aussi autour du classifieur dense , nombre de couches et de neurones ")
img_summary = Image.open('summary_modelDL.png')
st.image(img_summary,caption='sommaire du modèle multimodal DL')

st.write("visualisation graphique complémentaire qui a son intéret :")

img_plot_model = Image.open('keras_plot_model.png')
st.image(img_plot_model,caption='lien entre les blocs fonctionnels du modèle DL')

st.write("itérations essayées sur différentes valeurs de learning_rate et optimiseurs")

img_plot_model = Image.open('optimizer.png')
st.image(img_plot_model,caption='paramétrage pour la boucle de backpropagation')


st.write("le callback earlyStopping arrete l'entrainement si val_loss ne baisse plus")

img_training = Image.open('model_trainingDL.png')
st.image(img_training,caption="epochs exécutés pendant l'entrainement du modèle")

st.write("L'augmentation de la précision passe surement par un retour sur le préprocessing texte")


img_accuracy = Image.open('accuracy_modelDL.png')
st.image(img_accuracy,caption="évolution de la précision du modèle")

st.write("")

img_loss = Image.open('loss_modelDL.png')
st.image(img_loss,caption="évolution de la perte du modèle")

st.write("Résultats qui illustrent la capacité de classement de ce modéle DL en trés grande partie basée sur les features texte assurément, pour rappel un DL archi RNN GRU texte donne aussi 82 pour cent de précision")

img_report = Image.open('classification_reportDL.png')
st.image(img_report,caption="rapport de classification du modèle")

st.write("")

img_heatmap = Image.open('heatmapDL.png')
st.image(img_heatmap,caption="heatmap - matrice de confusion du modèle")

st.write("Les erreurs commises par le modèle sont en partie somme toute assez logiques au regard de la classification appliquée par Rakuten qui comporte vraiment une part de recouvrement entre certaines classes.")

img_errors = Image.open('model_errors.png')
st.image(img_errors,caption="erreurs et classes confondues par le modèle")


    









