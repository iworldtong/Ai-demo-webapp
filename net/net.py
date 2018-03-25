import random
import shutil
import os


# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image


# MODEL_PATH = './models/model1.h5'

# # Load trained model
# # model = load_model(MODEL_PATH)
# # model._make_predict_function()          # Necessary
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds




def gen_anime_profile(save_dir, model_path):

	# get effective image name
	random_name = str(random.randint(0, 1e5)) + '.jpg'
	while os.path.exists(os.path.join(save_dir, random_name)):
		random_name = str(random.randint(0, 10000)) + '.jpg'
	img_path = os.path.join(save_dir, random_name)

	# generate



	# temp 
	shutil.copyfile("static/img/1.jpg", img_path) 

	return img_path