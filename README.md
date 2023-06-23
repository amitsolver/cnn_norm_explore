# cnn_norm_explore
 Built image classification on CIFAR10 dataset and exploring various normalization techniques

 Target accuracy above 70% on test dataset with less than 50000 parameters using Batch Normalization , Layer Normalization and Group Normalization

 Experiments:

 | Normalization | Model Statistics | Analysis |
 | --- | --- | --- |
 | Network with Batch Normalization | <ul><li>Parameter : 18k</li><li>Train Accuracy : 74.8</li><li>Test Accuracy : 75.7</li></ul> | Faster Convergences initially and later converges slowly |
 | Network with Layer Normalization | <ul><li>Parameter : 23k</li><li>Train Accuracy : 72.1</li><li>Test Accuracy : 70.9</li></ul> | Converges slowly as compare to batch and group norm |
 | Network with Group Normalization | <ul><li>Parameter : 18k</li><li>Train Accuracy : 71.9</li><li>Test Accuracy : 72.5</li></ul> | Converges slowly as compared to bacth norm but faster than group norm |


 



