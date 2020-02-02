import os
import time
import shutil
import numpy as np
import tensorflow as tf
from utils import util as U
from utils.eval_saver import save_str, save_img

class Trainer:
    def __init__(self, model):
        self.model = model

    # @tf.function
    def train(self, 
            train_provider, 
            validation_provider,
            epochs, 
            batch_size, 
            output_path,
            optimizer=tf.keras.optimizers.Adam(), 
            mini_batch_size=None,
            eval_frequency=1,
            is_save_train_imgs=False,
            is_save_valid_imgs=True,
            is_rebuilt_path=True):

        if is_rebuilt_path and os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        iters = train_provider.size / batch_size
        assert iters > 0 and iters % 1 == 0, 'batch size {0} does not match the data size {1}.'.format(batch_size, train_provider.size)
        mini_batch_size = batch_size if mini_batch_size is None else mini_batch_size
        mini_iters = batch_size / mini_batch_size
        assert mini_iters > 0 and mini_iters % 1 == 0, 'mini batch size {0} does not match the batch size {1}.'.format(mini_batch_size, batch_size)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        ckpt = tf.train.Checkpoint(net=self.model.net)

        print('Start training: epochs {}, batch size {}, mini-batch size {}, training data {}, validation data {}.'
              .format(epochs, batch_size, mini_batch_size, train_provider.size, validation_provider.size))
              
        train_eval_str = {}
        valid_eval_str = {}
        time_start = time.time()
        for ep in range(epochs):
            ep_time_start = time.time()
            for _ in range(int(iters)):
                grads = None
                for _ in range(int(mini_iters)):
                    feed_dict = train_provider(mini_batch_size)
                    mini_grads = self.model.get_grads(feed_dict)
                    grads = self._grads_add(grads, mini_grads)
                grads = self._grads_div(grads, mini_iters)
                optimizer.apply_gradients(zip(grads, self.model.net.trainable_variables))
            ep_train_time = time.time() - ep_time_start
            ep_eval_time = 0
            if ep % eval_frequency == 0 or ep == epochs - 1:
                ep_train_eval = self.eval(train_provider, batch_size=mini_batch_size, print_str=False, need_imgs=is_save_train_imgs)
                ep_valid_eval = self.eval(validation_provider, batch_size=mini_batch_size, print_str=False, need_imgs=is_save_valid_imgs)
                ep_eval_time = time.time() - ep_train_time - ep_time_start
                if is_save_train_imgs:
                    save_img(ep_train_eval[1], '{}/train_imgs/'.format(output_path), ep)
                    ep_train_eval = ep_train_eval[0]
                if is_save_valid_imgs:
                    save_img(ep_valid_eval[1], '{}/valid_imgs/'.format(output_path), ep)
                    ep_valid_eval = ep_valid_eval[0]
                save_str(ep_train_eval, '{}/train_eval.txt'.format(output_path), ep)
                save_str(ep_valid_eval, '{}/valid_eval.txt'.format(output_path), ep)
                
                # time_ep_save_imgs_end = time.time()
            train_log = ('epoch {} ------ time cost: overall {:.1f} ------ step training {:.1f} ------ step evaluation {:.1f} ------'
                    .format(ep, time.time()-time_start, ep_train_time, ep_eval_time))
            
            if ep % eval_frequency == 0 or ep == epochs - 1:
                train_log += ('\n  train      : {}'.format(U.dict_to_str(ep_train_eval)) + \
                              '\n  validation : {}'.format(U.dict_to_str(ep_valid_eval)))

            print(train_log) 
            with open(output_path + '/train_log.txt', 'a+') as f:
                f.write(train_log + '\n')

            train_eval_str = U.dict_append(train_eval_str, ep_train_eval)
            valid_eval_str = U.dict_append(valid_eval_str, ep_valid_eval)
        
            # TODO add early stopping and best ckpt save
            # TODO add tensorboard summary
            ckpt.write(output_path + '/ckpt/final')
        

        return train_eval_str, valid_eval_str

    def restore(self, ckpt_path):
        ckpt = tf.train.Checkpoint(net=self.model.net)
        ckpt.restore(ckpt_path)
        

    def eval(self, data_in, **kwargs):
        batch_size = kwargs.get('batch_size', 0)
        print_str = kwargs.get('print_str', True)
        need_imgs = kwargs.get('need_imgs', False)
        print_all_results = kwargs.get('print_all_results', False)  # to print each results and index
        if type(data_in) is dict:
            time_start = time.time()
            eval_dict = self.model.eval(data_in, **kwargs)
            time_cost = time.time() - time_start
            if print_str:
                print('Evaluation time cost is {:.1f}'.format(time_cost))
            return eval_dict
        
        data_provider = data_in
        ndata = data_provider.size
        m = ndata // batch_size
        n = ndata % batch_size
        eval_dict_str = {}
        eval_dict_img = {}
        time_start = time.time()
        for _ in range(m):
            data_dict = data_provider(batch_size)
            sub_eval_dict_str, sub_eval_dict_img = self.model.eval(data_dict, **kwargs)
            eval_dict_str = U.dict_concat(eval_dict_str, sub_eval_dict_str)
            if need_imgs:
                eval_dict_img = U.dict_append(eval_dict_img, sub_eval_dict_img)
        if n > 0:
            sub_eval_dict_str, sub_eval_dict_img = self.model.eval(data_provider(n), **kwargs)
            eval_dict_str = U.dict_concat(eval_dict_str, sub_eval_dict_str)
            if need_imgs:
                eval_dict_img = U.dict_append(eval_dict_img, sub_eval_dict_img)
        if print_str:
            time_cost = time.time() - time_start
            print('Evaluate {} data, time cost is {:.1f}'.format(ndata, time_cost))
            print('  {}'.format(U.dict_to_str(eval_dict_str)))
        if print_all_results:
            results_dict = {}
            for key in eval_dict_str:
                value = eval_dict_str.get(key)
                if value.shape[0] == ndata:
                    results_dict[key] = value
            for i in range(ndata):
                print_str = "Picture Index: {}\t".format(i)
                for key in results_dict:
                    print_str += "{}: ".format(key)
                    for k in results_dict[key][i]:
                        print_str += "{:.5f} ".format(k)
                    print_str += "\t"
                print(print_str)
        if need_imgs:
            return eval_dict_str, eval_dict_img
        else:
            return eval_dict_str

    def predict(self, data_dict):
        return self.model.predict(data_dict)

    def _grads_add(self, grads, mini_grads):
        if grads is None:
            grads = mini_grads
        else:
            for i, g in enumerate(mini_grads):
                if g is not None:
                    grads[i] += g
        return grads
    
    def _grads_div(self, grads, n):
        if n != 1 :
            for i in range(len(grads)):
                if grads[i] is not None:
                    grads[i] /= n
        return grads