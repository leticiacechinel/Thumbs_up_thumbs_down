import streamlit as st
from tensorflow import keras
import os
import cv2
import numpy as np

model = keras.models.load_model('modelo')


button = st.button("Processar imagem")

class ConfigSt():
    def __init__(self):
    	self.main_folder = "./val/"


    def select_files(self, path):
        files = []
        files = [file for file in os.listdir(path)]
        files.insert(0,'')
        selected_file = st.selectbox("Escolha um Arquivo:", files)

        return selected_file

    def select_folder(self):
        folder_path = self.main_folder
        dirname = [di for di in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, di))]
        dirname.insert(0,'')
        selected_path = st.selectbox("Escolha um Diretório 📁:", dirname)
        return os.path.join(folder_path, selected_path)



if __name__ == '__main__':
    conf = ConfigSt()
    path = conf.select_folder()
    if path:
    	file = conf.select_files(path)


    if button:	
        img = cv2.imread(path + '/' + file)  
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),width=200)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        resized = cv2.resize(gray, (100, 100)).astype('float32')
        resized = resized.reshape(100,100,1)
        resized = np.expand_dims(resized, axis=0)
        teste = resized/255

        result = model.predict_classes(teste)
        if result == 1:
            result = "thumbs up!"
        if result == 0:
        	result = "thumbs down!"

        st.markdown("A classe eh " + result)