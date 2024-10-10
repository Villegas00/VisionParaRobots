#librerias
import cv2
import numpy as np
import time
import easyocr
import re
from gtts import gTTS
from pybraille import convertText
from PIL import Image
from lavis.models import load_model_and_preprocess
from googletrans import Translator

#tamaño de los frames
IMG_ROW_RES = 480
IMG_COL_RES = 640

#funciones camara
def init_camera():
	video_capture = cv2.VideoCapture(0)
	ret = video_capture.set(3, IMG_COL_RES)
	ret = video_capture.set(4, IMG_ROW_RES)
	return video_capture

def acquire_image(video_capture):
	ret, frame = video_capture.read()
	scaled_rgb_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
	scaled_rgb_frame = scaled_rgb_frame[:, :, ::-1]
	return frame, scaled_rgb_frame

def show_frame(name, frame):
	cv2.imshow(name, frame)

#funcion audio
def tts(text_file, lang, name_file):
    with open(text_file, "r") as file:
        text = file.read()
    file = gTTS(text = text, lang = lang)
    filename = name_file
    file.save(filename)

#funcion traductor
def translate_text(texto):
    translator = Translator(service_urls=['translate.google.com'])
    translated = translator.translate(texto, dest='es')
    return translated.text

#Modelo para detección de textos
reader = easyocr.Reader(["es"])

#Modelo para descripcion de imagenes
device = "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device)

#tiempos para comunicación
lastPublication = 0.0
PUBLISH_TIME =10

#iniciando percepción
video_capture = init_camera()

#bucle de percepción
while(True):
	#Capa de senseo
	bgr_frame, scaled_rgb_frame = acquire_image(video_capture)

	#Capa de comunicación
	if np.abs(time.time()-lastPublication) > PUBLISH_TIME:
		try:
			print("No remote action needed ....")
		except (keyboardInterrupt):
			break
		except Exception as e:
			print(e)
		lastPublication = time.time()

	# Añadir nuestro codigo aqui 
	#Convertir imagen en blanco y negro. 

	gris = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
 
	# Aplicar suavizado Gaussiano
	gauss = cv2.GaussianBlur(gris, (5,5), 0)
	

	#Apretar tecla 'f' para funcion.
	
	if cv2.waitKey(1) & 0xFF == ord('f'):
		#imagen para deteccion de texto
		img_text= gauss

		#imagen para descripcion de escena
		img = Image.fromarray(img_text)
		raw_image = img.convert("RGB")

		# First preprocess the input image
		image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		# Generate caption, for this example 3 different captions are requested
		img_description=model.generate({"image": image}, use_nucleus_sampling=True, num_captions=5)

		#Traducir la descripcion dada en ingles a español
		description ='. '.join(img_description) # por default viene en inglés
		english_text = description
		spanish_text = translate_text(english_text)
		print("Three instances of the descriptive text: \n", spanish_text)
		
		#Escribir la traducción en  un archivo de texto para despues convertirlo a audio
		d = open ('Descripcion.txt','w', encoding='utf-8')
		d.write(spanish_text + '\n')
		d.close()
		tts("Descripcion.txt", "es", "audio_descripcion.mp3")
		
		#Procesamiento de la imagen con modelo easyocr
		result = reader.readtext(gauss, paragraph=True)

		#Si existe algun texto
		if(result!=[]):
		#Abrir archivos de texto 	
		  f = open ('Texto en Braille.txt','w', encoding='utf-8')
		  s = open ('Texto Imagen.txt','w')
		  #Bucle para obtener resultados
		  for res in result:
		  	#definir puntos del texto
		    pt0 = res[0][0]
		    pt1 = res[0][1]
		    pt2 = res[0][2]
		    pt3 = res[0][3]
		    #Crear cuadro e insertar texto obtenido
		    cv2.rectangle(gauss, pt0, (pt1[0], pt1[1] - 23), (166, 56, 242), 2)
		    cv2.putText(gauss, res[1], (pt0[0], pt0[1] -3), 2, 0.8, (255, 255, 255), 1)
		    #Encuadrar texto
		    cv2.rectangle(gauss, pt0, pt2, (166, 56, 242), 2)
		    cv2.circle(gauss, pt0, 2, (0, 0, 0), 2)
		    cv2.circle(gauss, pt1, 2, (0, 0, 0), 2)
		    cv2.circle(gauss, pt2, 2, (0, 0, 0), 2)
		    cv2.circle(gauss, pt3, 2, (0, 0, 0), 2)
		    #Procesamiento de string obtenido
		    strmod= re.sub(r'[^A-Za-z0-9 ]+', '', res[1])
		    #Convertir texto a braille
		    b_text=convertText(strmod)
		    #Escribir texto en español y braille en un archivo de texto
		    f.write(b_text + '\n')
		    s.write(strmod + '\n')
		  f.close()
		  s.close()
		  #Convertir a audio el texto en español
		  tts("Texto Imagen.txt", "es", "audio_texto.mp3")
		#En caso de no tener texto la imagen
		else:
		#Abir archivo de texto, escribir que no hay un texto en la escena y convertirlo en audio
		  s = open ('Texto Imagen.txt','w')
		  s.write("No hay texto legible en la escena " + '\n')
		  s.close()
		  tts("Texto Imagen.txt", "es", "sonido_generado.mp3")
		#Mostrar imagen con encuadres de texto  
		show_frame('Texto imagen', img_text)

	#Capa de visualización	
	show_frame('RGB image', gauss)
	

	#Apretar tecla 'e' para salir de programa
	if cv2.waitKey(1) & 0xFF == ord('e'):
		break

video_capture.release()