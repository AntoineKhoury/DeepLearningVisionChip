#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def concatenate_4to1 (A,B,C,D,size):
    #Need 4 tensors of the same dimenssions to map each of their values
    #in a 4 times bigger feature map, need to be same size
    #size is the size of each image with total image is 4*size*size dimenssion
    
    flatten_A= tf.reshape(A,[-1,1])
    flatten_B= tf.reshape(B,[-1,1])
    flatten_C= tf.reshape(C,[-1,1])
    flatten_D= tf.reshape(D,[-1,1])
    
    concat_AB= tf.concat([flatten_A,flatten_B],1)
    concat_CD= tf.concat([flatten_C,flatten_D],1)
    
    reshaped_AB= tf.reshape(concat_AB,[-1,size,2*size,1])
    reshaped_CD= tf.reshape(concat_CD,[-1,size,2*size,1])
    
    concat_ABCD= tf.concat([reshaped_AB,reshaped_CD],1)
    return tf.reshape(concat_ABCD,[-1,size*2,size*2,1])

t1 =[[1,2],[5,6]]
t2 =[[7,8],[11,12]]
t3 =[[13,14],[17,18]]
t4 =[[19,20],[23,24]]

combined= concatenate_4to1(t1,t2,t3,t4,2)