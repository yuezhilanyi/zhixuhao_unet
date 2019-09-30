import os
import sys
import gdal
import tqdm
import argparse

import numpy as np

from termcolor import colored

from model import unet


def main(args, input_lst):
    padded_isize = 256
    pad = 64
    isize = 256

    # restoring models
    model = unet(input_size=(256, 256, 6))
    # loading weights
    weight_path = 'unet_change.hdf5'
    model.load_weights(weight_path)
    print(colored('Done restoring unet model from {}'.format(weight_path)))

    for sub_lst in input_lst:
        try:
            tqdm.tqdm.write('Processing: {}, {}'.format(sub_lst[0], sub_lst[1]), file=sys.stderr)

            # 为了支持中文路径，请添加下面这句代码
            gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

            src_ds_1 = gdal.Open(sub_lst[0], gdal.GA_ReadOnly)  # 只读方式打开原始影像
            src_ds_2 = gdal.Open(sub_lst[1], gdal.GA_ReadOnly)  # 只读方式打开原始影像
            if src_ds_1 is None or src_ds_2 is None:
                print('Unable to open %s or %s' % (sub_lst[0], sub_lst[1]))
                sys.exit(1)

            geoTrans = src_ds_1.GetGeoTransform()  # 获取地理参考6参数
            srcPro = src_ds_1.GetProjection()  # 获取坐标引用
            srcXSize = src_ds_1.RasterXSize  # 宽度
            srcYSize = src_ds_1.RasterYSize  # 高度
            nbands = src_ds_1.RasterCount  # 波段数
            new_geoTrans = (geoTrans[0] + geoTrans[1] * pad,
                            geoTrans[1], geoTrans[2],
                            geoTrans[3] + geoTrans[5] * pad,
                            geoTrans[4], geoTrans[5])

            driver = gdal.GetDriverByName('GTiff')
            raster_fn = sub_lst[0][0:-4] + '_mask_unet_{}.tif'.format(str(isize))
            outRaster = driver.Create(raster_fn, srcXSize - pad * 2, srcYSize - pad * 2, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])
            outRaster.SetGeoTransform(new_geoTrans)
            outRaster.SetProjection(srcPro)
            outband = outRaster.GetRasterBand(1)

            for m in tqdm.trange(0, srcYSize - pad * 2, isize):
                if m + isize > srcYSize - pad * 2:
                    m = srcYSize - pad * 2 - isize

                for n in range(0, srcXSize - pad * 2, isize):
                    if n + isize > srcXSize - pad * 2:
                        n = srcXSize - pad * 2 - isize

                    dsa1 = src_ds_1.ReadAsArray(n, m, padded_isize, padded_isize)
                    dsa2 = src_ds_2.ReadAsArray(n, m, padded_isize, padded_isize)
                    crop = np.concatenate((dsa1, dsa2))
                    im = np.rollaxis(crop, 0, 3)
                    im = np.array(im, dtype=np.float32)
                    im /= 255.0
                    im = np.expand_dims(im, axis=0)

                    # Take the edge map from the network from side layers and fuse layer
                    result = model.predict(im, batch_size=1, verbose=0)
                    result = np.squeeze(np.array(result * 255).astype(np.uint8))
                    out_array = np.squeeze(result)[pad: -pad, pad: -pad]
                    outband.WriteArray(out_array, n, m)
            outRaster.FlushCache()
        except ValueError as e:
            tqdm.tqdm.write("ValueError: ".format(e), file=sys.stderr)
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility for inference')
    args = parser.parse_args()

    input_lst = [['../images/201803bj_changpingWGS84.img', '../images/201805_changpingWGS84.img'],
                 ['../images/201803bj_tongzhouWGS84.img', '../images/201805_tongzhouWGS84.img']]

    # train_data_dir = '../dataset/train_data_v5'
    # train_data_lst = [os.path.join(train_data_dir, e) for e in os.listdir(train_data_dir) if 'unet' not in e]
    # input_lst = []
    # for i in range(0, len(train_data_lst), 3):
    #     input_lst.append([train_data_lst[i], train_data_lst[i+1]])
    # print(input_lst)

    main(args, input_lst)

