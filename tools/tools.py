import base64 
import os



def return_img_stream(img_path, remove_later=False):   
    if not os.path.exists(img_path):
        img_path = 'static/img/default.jpg'

    with open(img_path, 'rb') as img_f:  
        img_stream = img_f.read()          
        img_stream = base64.b64encode(img_stream)  
    if remove_later: # to save memory        
        os.remove(img_path)
    return img_stream  