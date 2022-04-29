import numpy as np
import os
from scipy import stats
import cv2

def get_label(document_path1, document_path2):
	#Aqui eu vejo quem vem antes do / no nome do arquivo
	#Se for a mesma string, e a mesma impressora (retorno 1)
	#Caso contrario, e outra impressora (retorno 0)
	
	printer_name1=document_path1.split('/')[0]
	printer_name2=document_path2.split('/')[0]
	
	print(printer_name1)
	print(printer_name2)

	if(printer_name1==printer_name2):
		return 1
	else:
		return 0

def make_prediction(document_path1, document_path2, model, threshold):
	#Aqui eu faco a previsao
	#Para aquele par de documentos, eu pego o primeiro do primeiro documento e comparo com o primeiro do segundo documento
	#E assim por diante
	directory_document_1="/media/anselmo/Volume/Dataset_Open_Set/Blocks-Variance/64/"+document_path1+"/"
	directory_document_2="/media/anselmo/Volume/Dataset_Open_Set/Blocks-Variance/64/"+document_path2+"/"
	
	
	blocks_document_1=np.array(os.listdir(directory_document_1))
	blocks_document_2=np.array(os.listdir(directory_document_2))
		    
	#Predicao para cada par
	predictions_for_that_pair=[]
	
	#Vamos comecar a predicao bloco correspondente a bloco correspondente
	for i in range(0,blocks_document_1.shape[0]):
		block_1_path= directory_document_1 + '/' + blocks_document_1[i]
		block_2_path= directory_document_2 + '/' + blocks_document_2[i]
		
		#Vamos ler os pares, preparar para o teste e testar
		#Lendo par (imagem1)
		pair_1=cv2.imread(block_1_path)
		#Preparando par (imagem1)
		pair_1 = np.expand_dims(pair_1, axis=-1)
		pair_1 = np.expand_dims(pair_1, axis=0)
		pair_1 = pair_1 / 255.0
    		
		#Lendo par (imagem2)
		pair_2=cv2.imread(block_2_path)
		#Preparando par (imagem1)
		pair_2 = np.expand_dims(pair_2, axis=-1)
		pair_2 = np.expand_dims(pair_2, axis=0)
		pair_2 = pair_2 / 255.0
    		
		#Testando o par
		predictions_for_those_blocks = model.predict([pair_1, pair_2])
		
		#The sigmoid activation function is used here because the output range of the function is [0, 1]. 
		#An output closer to 0 implies that the image pairs are less similar (and therefore from different classes), 
		#while a value closer to 1 implies they are more similar (and more likely to be from the same class).		
		if (predictions_for_those_blocks[0][0]<threshold):
			prediction=1
		else:
			prediction=0
			
		
		predictions_for_that_pair.append(prediction)
	
	predictions_for_that_pair=np.array(predictions_for_that_pair)
	
	#vejamos a moda
	vals,counts = np.unique(predictions_for_that_pair, return_counts=True)
	index = np.argmax(counts)
	mode=vals[index]

	#retorno a classe mais votada (1-mesmo, 0-diferente)
	return(mode)
	
def make_prediction_only(document_path1, document_path2, block_size, model):
	#Aqui eu faco a previsao
	#Para aquele par de documentos, eu pego o primeiro do primeiro documento e comparo com o primeiro do segundo documento
	#E assim por diante
	directory_document_1=document_path1+"/"
	directory_document_2=document_path2+"/"
	
	blocks_document_1=np.array(os.listdir(directory_document_1))
	blocks_document_2=np.array(os.listdir(directory_document_2))
		    
	#Predicao para cada par
	predictions_for_pairs=[]
	labels_for_pairs=[]
	
	#Vamos comecar a predicao bloco correspondente a bloco correspondente
	for i in range(0,blocks_document_1.shape[0]):
		block_1_path= directory_document_1 + '/' + blocks_document_1[i]
		block_2_path= directory_document_2 + '/' + blocks_document_2[i]
		
		#Vamos ler os pares, preparar para o teste e testar
		#Lendo par (imagem1)
		pair_1=cv2.imread(block_1_path)
		#Preparando par (imagem1)
		pair_1 = np.expand_dims(pair_1, axis=-1)
		pair_1 = np.expand_dims(pair_1, axis=0)
		pair_1 = pair_1 / 255.0
    		
		#Lendo par (imagem2)
		pair_2=cv2.imread(block_2_path)
		#Preparando par (imagem1)
		pair_2 = np.expand_dims(pair_2, axis=-1)
		pair_2 = np.expand_dims(pair_2, axis=0)
		pair_2 = pair_2 / 255.0
		
		label=get_label(document_path1, document_path2)
		labels_for_pairs.append(label)    		
		#Testando o par
		prediction = model.predict([pair_1, pair_2])
		predictions_for_pairs.append(prediction)
		#print(len(predictions_for_pairs))
		#input()
		
	predictions_for_pairs=np.array(predictions_for_pairs)
	labels_for_pairs=np.array(labels_for_pairs)
	

	predictions_for_pairs=np.squeeze(predictions_for_pairs)
	#print(predictions_for_pairs.shape)
	#print(labels_for_pairs.shape)
	
	#input()
	#retorno a classe mais votada (1-mesmo, 0-diferente)
	return(predictions_for_pairs,labels_for_pairs)
    		
		
	
	










