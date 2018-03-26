import random
import shutil
import os


# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

import net




def gen_anime_profile(save_dir):
	netd_path = 'models/gan_anime_model/netd_200.pth'
	netg_path = 'models/gan_anime_model/netg_200.pth'
	if not os.path.exists(netd_path): netd_path=None
	if not os.path.exists(netg_path): netg_path=None

	# get effective image name
	random_name = str(random.randint(0, 1e5)) + '.jpg'
	while os.path.exists(os.path.join(save_dir, random_name)):
		random_name = str(random.randint(0, 10000)) + '.jpg'
	img_path = os.path.join(save_dir, random_name)

	# generate images
	kwargs = {
			'netd_path': netd_path,
			'netg_path': netg_path,
			'save_path': img_path
		}
	net.gan_anime_model.generate(**kwargs)

	return img_path