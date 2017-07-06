Wrote binary convolutional and fully-connected layers
 - Didn't tf.sign() the kernel
 - No scaling factor

"2bin" - First three layers are binary, last two layers are fp16
Loss 32.66 => 2.42 in 9000 steps
=> 9.79 in 25000 steps, => 911.60 in 40000 steps, => 152.12 in 50000 steps
precision@1 = 0.100 (phase=False (wrong) during test phase...)

"arch1" - First two layers are fp16, last three layers of binary
Loss 20.82 => 19.03 in 38000 steps
precision@1 = 0.119 (phase=False (wrong) on test phase...)

"arch2" - First four layers are binary, last layer is fp16
Loss 644.81 => 364.06 in 65000 steps
precision@1 = 0.302 (phase=False (wrong) on test phase...)

"arch3" - First two layers are fp16, middle two layers are binary, last layer is fp16
Loss 2365.00 => 670.02 in 25000 steps
precision@1 = 0.277 (phase=False (wrong on test phase...)

Benchmark - all 5 layers are fp16
Loss 4.68 => 0.13 in 350000 steps
precision@1 = 0.866
