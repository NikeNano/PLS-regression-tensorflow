# PLS-regerssion-tensorflow
PLS regression in TensorFlow 2.0 


To do:
- Look over the structure and ensure we have a tensorflow approach
- Make the code follow the graph structure of tensorflow 
- Replace the use of lists to tensorflow operations instead
- Implement logging of the traning and paramters in tensorboard
- Update performance test, today it is not complete. Should we look at GPU, tests aswell
- Update all functions to use tf.functions
- Investigate the need for implementing numeric solution as option. Can we build a gradient descent solution for PLS? 
- Should we look at tensorflow serving with our model to test it out? 
- Implement test(pytest?) for all functions
- Make sure that the PLS can be saved and loaded correctly, should we use tf.savedmodel or pickle? 


Note: Tensorflow version used is 2.0.0rc0
