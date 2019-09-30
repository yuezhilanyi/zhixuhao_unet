from model import *
from data import *
import cv2
from math import ceil

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 8
class_weight = [650]  # or 'auto
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(batch_size, '../dataset/for_unet/train', 'image', 'label', data_gen_args,
                        image_color_mode='rgbrgb', save_to_dir="../dataset/for_unet/train/aug")
num_samples = len(os.listdir('../dataset/for_unet/train/image'))

model = unet(input_size=(256, 256, 6))
model_checkpoint = ModelCheckpoint('unet_change.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene,
                    steps_per_epoch=ceil(num_samples/batch_size),
                    epochs=30,
                    callbacks=[model_checkpoint],
                    class_weight=class_weight)

# restoring models
model = unet(input_size=(256, 256, 6))
# loading weights
model.load_weights('unet_change.hdf5')

test_dir_path = "../dataset/for_unet/test"
lst = [e for e in os.listdir(test_dir_path) if 'predict' not in e]
for fn in lst:
    fn = os.path.join(test_dir_path, fn)
    im = gdal.Open(fn).ReadAsArray()
    im = np.rollaxis(im, 0, 3)
    im = np.array(im, dtype=np.float32)
    im /= 255.0
    im = np.expand_dims(im, axis=0)

    # Take the edge map from the network from side layers and fuse layer
    result = model.predict(im, batch_size=1, verbose=0)
    result = np.squeeze(np.array(result * 255).astype(np.uint8))
    tar_fn = "{}_predict.png".format(fn[:-4])
    cv2.imwrite(tar_fn, result)

# testGene = testGenerator("../dataset/for_unet/test", num_image=398)
# results = model.predict_generator(testGene, 398, verbose=1)
# save_path = "../dataset/for_unet/test"
# for i in range(results.shape[0]):
#     im = results[i]
#     cv2.imwrite(os.path.join(save_path, "%d_predict.png" % i), (im*255).astype('uint8'))
# saveResult("../dataset/for_unet/test", results)
