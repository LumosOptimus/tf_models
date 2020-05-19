# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.util import deprecation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd

from functools import partial
import warnings

deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.filterwarnings('ignore')



class PreprocessingData(object):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.cat_features = list(self.X.columns.get_indexer(self.X.select_dtypes('object').columns))
        self.num_features = list(self.X.columns.get_indexer(self.X.select_dtypes('int').columns)) \
                          + list(self.X.columns.get_indexer(self.X.select_dtypes('float').columns))
    
    
      
    def feature_encoding(self):
        '''
        Encode preprocessed dataframe for DNN
        LabelEncoding -> OneHotEncoding -> concat(df)
        
        args:
            X          - train features from feature_selection func
            
        return:
            encoded_df - {DataFrame} filled with ohe encoded features
        '''
        print('Encode X features...')
        X = self.X
        init_features = X.iloc[:, self.num_features].reset_index(drop = True)
        encoded_features = []

        for i in self.cat_features:
            le = LabelEncoder()
            X.iloc[:, i] = le.fit_transform(X.iloc[:, i])
            
            ohe = OneHotEncoder(handle_unknown = 'ignore')
            ohe_arr = ohe.fit_transform(X[[X.columns[i]]]).toarray()
           
            names = le.classes_
            this_feature_df = pd.DataFrame(ohe_arr, columns = [str(X.columns[i]) + '_' +  str(cat_name) for cat_name in names])
            encoded_features.append(this_feature_df)
        
        try: 
            encoded_features = pd.concat(encoded_features, axis = 1).reset_index(drop = True)
            print('features encoding done \nencoded features data shape:', encoded_features.shape)
            total_train_data = pd.concat([init_features, encoded_features], axis = 1)
            print('totol train data shape:', total_train_data.shape)
            return total_train_data
        except ValueError as e:
            print(e)
       
        
        
    def feature_scaling(self, X_encoded, X_train, X_test):
        '''
        Standardize features by removing the mean and scaling to unit variance
        
        args:
            X_encoded    - 
            X_train      - 
            X_test       - 
            
        return:
            train_scaled - encoded X_train but scaled
            test_scaled  - encoded X_test but scaled
        '''
        scaler = StandardScaler()
        scaler.fit(X_encoded, self.y)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)

        return train_scaled, test_scaled



class ModelDNN(object):
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_inputs = self.X_train.shape[1]
        self.n_hidden1 = np.ceil(self.n_inputs / 2)
        self.n_hidden2 = np.ceil(self.n_hidden1 / 2)
        self.n_hidden3 = np.ceil(self.n_hidden2 / 2)
        self.n_outputs = 1
        self.n_epochs = 330
        self.batch_size = 50
        self.learning_rate = 0.0001
        
        
    def run_model(self):
        
        X = tf.placeholder(tf.float32, shape = [None, self.n_inputs], name = 'X')
        y = tf.placeholder(tf.float32, shape = [None], name = 'y')
        training = tf.placeholder_with_default(False, shape = (), name = 'training')       
        batch_norm_layer = partial(tf.layers.batch_normalization, training = training, momentum = 0.9)
        
        with tf.name_scope('dnn'):
            hidden1 = tf.layers.dense(X, self.n_hidden1, activation = tf.nn.elu, name = 'hidden1')
            batch_norm1 = batch_norm_layer(hidden1)
            hidden2 = tf.layers.dense(batch_norm1, self.n_hidden2, activation = tf.nn.elu, name = 'hidden2')
            batch_norm2 = batch_norm_layer(hidden2)
            hidden3 = tf.layers.dense(batch_norm2, self.n_hidden3, activation = tf.nn.elu, name = 'hidden3')
            batch_norm3 = batch_norm_layer(hidden3)
            hidden4 = tf.layers.dense(batch_norm3, self.n_hidden3, activation = tf.nn.elu, name = 'hidden4')
            batch_norm4 = batch_norm_layer(hidden4)
            output_before_bn = tf.layers.dense(batch_norm4, self.n_outputs, name = 'outputs')
            output = batch_norm_layer(output_before_bn)
    
        with tf.name_scope('loss'):
            rmse = tf.squared_difference(y, output)
            loss = tf.reduce_mean(rmse, name= 'loss')
            
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            training_op = optimizer.minimize(loss)
                                             
        with tf.name_scope('eval'):
            error = tf.multiply(tf.abs(tf.divide(tf.subtract(y, output), y)), 100)
            mape = tf.reduce_mean(error)
        
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for idx, batch in enumerate(range(0, self.X_train.shape[0], self.batch_size)):
                    X_batch, y_batch = self.X_train[batch: batch + self.batch_size], self.y_train[batch: batch + self.batch_size]
                    sess.run([training_op, extra_update_ops], 
                             feed_dict = {
                                 X: X_batch, 
                                 y: y_batch
                                 })
                accuracy_val = mape.eval(feed_dict = {
                    X: self.X_test, 
                    y: self.y_test
                    })
                print('epoch: {}, MAPE: {}'.format(epoch, accuracy_val))
            save_path = saver.save(sess, 'tf_models/tf_bn_model.ckpt')
            
        return 
