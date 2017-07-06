Updated/fixed binary convolutional and fully-connected layers
 - tf.sign() the kernel
 - Including scaling factor (L1 Norm / n)

"3bin" - Second layer (convolutional) is binary, rest are fp16
Loss 4.67 => 1.28 in 30000 steps
precision@1 = 0.761

"3arch1" - Middle three layers are binary, rest are fp16

"3arch2" - Middle two fully-connected layers are binary, rest are fp16

"3arch3" - Middle two (one convolutional, one fully-connected) layers are binary, rest are fp16

