# PLS-regerssion-tensorflow
PLS regression implementation for  TensorFlow 2.0 


## Development setup

Install last release candidate for TensorFlow 2.0
`pip install tensorflow==2.0.0-rc0`

## TODOs

* Look over the structure and ensure we have a tensorflow approach
* Make the code follow the graph structure of tensorflow 
* Replace the use of lists to tensorflow operations instead
* Implement logging of the traning and paramters in tensorboard
* Update performance test, today it is not complete. Should we look at GPU, tests aswell
* Update all functions to use tf.functions
* Investigate the need for implementing numeric solution as option. Can we build a gradient descent solution for PLS? 
* Should we look at tensorflow serving with our model to test it out? 
* Implement test(pytest?) for all functions
* Make sure that the PLS can be saved and loaded correctly, should we use tf.savedmodel or pickle? 
* Add more regression algorithms

## Release History
* 0.0.1
    * PLS basic object working. More functionality needs to be added

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.


## Contributors
[https://github.com/NikeNano](https://github.com/NikeNano)
[https://github.com/jiwidi](https://github.com/jiwidi)
## Contributing

1. Fork it (<https://github.com/jiwidi/PLS-regerssion-tensorflow/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


