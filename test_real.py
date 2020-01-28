import tensorflow as tf
import numpy as np
import glob
import os
from argparse import ArgumentParser
import scipy.io as sio
import model

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--gpu', dest='gpu',default='0')
    parser.add_argument('--datapath', dest='datapath', default='LR_mat(x2)')
    parser.add_argument('--modelpath', dest='modelpath',default='Model_B')

    return parser

parser = build_parser()
option = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = option.gpu

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.9

'''Dataset'''
Sets=['Test','Part1','Part2','Part3']
ind=0

def main():
    scale = 2
    model_type='REF'

    datapath=option.datapath
    modelpath=option.modelpath

    img_list=np.sort(np.asarray(glob.glob(os.path.join(datapath,Sets[ind],'*.mat'))))

    fileNum=len(img_list)
    print(img_list)

    with tf.Session(config=conf) as sess:

        for n in range(fileNum):
            image = sio.loadmat(img_list[n])

            input_REF = image['REF']
            input_REF = np.tanh(input_REF)

            [HH, WW]= np.shape(input_REF)
            input_img = tf.placeholder(tf.float32, [1, HH, WW, 1])

            testCNN = model.REF_Network(input_img, scale, reuse=tf.AUTO_REUSE)
            output = testCNN.output
            saver = tf.train.Saver()

            ckpt_model = os.path.join(modelpath, 'model')

            print(ckpt_model, os.path.basename(img_list[n]))
            saver.restore(sess, ckpt_model)

            img=input_REF[None,:,:,None]
            out=sess.run(output,feed_dict={input_img: img})

            savefolder='REF_result'

            if not os.path.exists('%s/%s' % (savefolder,Sets[ind])):
                os.makedirs('%s/%s' % (savefolder,Sets[ind]))

            sio.savemat(os.path.join('%s/%s' % (savefolder, Sets[ind]), os.path.basename(img_list[n][:-4]+'_%s.mat' % model_type)),{model_type: out[0,:,:,0]})


if __name__=='__main__':
    main()
    print('Done')