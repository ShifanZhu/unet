from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# sess = tf.Session(config=tf.ConfgiProto(log_device_placement=True))
import tensorflow as tf
print("tf gpu:", tf.test.is_gpu_available())

model = unet(pretrained_weights="unet_membrane.hdf5")

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,20,verbose=1)
saveResult("data/membrane/test",results)