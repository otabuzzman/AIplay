# AIplay
A lab for playing around with AI concepts. The lab intends to gain some practise in implementing running programs of straight forward examples for Neural Networks and Deep Learning.

## MYONN
A SwiftUI implementation of Make Your Own Neural Network (MYONN). The app downloads the MNIST dataset and enables training and prediction using the files included in the set. A sketch area accepts handwritten input for prediction. There are save and load functions for trained models in a simple and convenient format.

The app features stochastic gradient descent (SGD) and mini-batch which is the default. Batch size is a hard-coded value of 30. To make use of SGD set `miniBatchSize` in `NetworkView.swift` to 1. Mini-batch makes use of any available cores. To enable Metal compute shader for the activation function set `tryOnGpu` in `Network.swift` to `true`.

The hard-coded network configuration is 784/ 100/ 10 fully-connected input/ hidden/ output nodes with sigmoid activation.

### Usage
On the first start the app asks for a folder to store the MNIST dataset. The four MNIST status indicators turn into green when all files have been downloaded and unzipped. Training the whole dataset took between 30 (mini-batch) and 45 (SGD) minutes on an iPad Pro 2022.

Input from the sketch area must be centered with some padding to the edges. Ideally, the images in the _Query input_ and _Query image_ areas will look quite similar.

![](Assets.xcassets/screenshot.imageset/Image Asset.png)

1. Basic information and usage (this text).
2. Progress indicator.
3. Query with random image from MNIST test set (contains 10000 items).
4. Prediction result. Turns red to indicate false prediction for test set item.
5. Display raw image of query input from dataset or handwritten.
6. Query with handwritten number input. Experimental feature. Input position in the sketch area and line width must correspond to MNIST. To get an idea have a look at some images (5) when performing random MNIST test set queries (3). To increase line width on larger displays (e.g. iPad or device held landscape) try drawing input multiple times.
7. Clear sketch area.
8. Reload the MNIST dataset into memory. Downloads the files and unzips them if necessary. Saves files in the app folder selected on first launch.
9. Per dataset-file state. Grey: present, yellow: loading, red: error and green: success. Circles from left to right correspond to files containing training images, training labels, test images and test labels respectively.
10. Mini-batch size (hard-coded).
11. Learning rate (hard-coded).
12. Number of trainings with full training set (contains 60000 items) applied to current network.
13. Time required to train the most recent epoch.
14. Train network with next mini-batch from MNIST training set.
15. Train network with full MNIST training set. Takes a couple of minutes. 
16. Query network with full MNIST test set and calculate accuracy.
17. Network accuracy. Multiply by 100 to get percent.
18. Save current network (model) in Files app in a roprietary format.
19. Load network with a model from the Files app. Overwrites current network without warning.
20. Reset network without warning. Stops training in progress. 

**Working**
- MNIST loading/ training/ predicting
- Save/ load trained model (proprietary format)
- Predict handwritten input

**Wanted**
- Configuration UI
- Save/ load in common format
- CoreML usage
- Metal compute shader usage (extended)
- More activation functions
- More layer types

### Tools
Apps used on iPad
- [Swift Playgrounds 4](https://apps.apple.com/de/app/swift-playgrounds/id908519492) (SP4)
- [Working Copy](https://workingcopyapp.com/) (WC)
- [Textastic](https://www.textasticapp.com/)
- [GitHub](https://apps.apple.com/us/app/github/id1477376905)

### Build
- Create and open a new app in SP4
- Delete predefined `*.swift` files
- Copy files (except Git stuff) from repository:
  - Get repository on iPad (Working Copy)
  - Copy from WC to SP4 (Textastic)
- Add [Playgrounds tester package](https://github.com/Losiowaty/PlaygroundTester.git)
- Add [Data compression package](https://github.com/mw99/DataCompression.git)

### Which file for what
|File|Comment|
|:---|:------|
|AIplay.swift|The main program.|
|CustomCoder.swift|An encode/ decode implementation.|
|FolderPicker.swift|A view. The function is in the name.|
|GaussianDistribution.swift|A class to provide random floats with normal distribution.|
|Matrix.swift|A somewhat pimped Array for matrix operations. Leverages vDSP.|
|MNISTDatasetView.swift|MVVM to put MNIST on disk.|
|MYONNView.swift|The top-level view.|
|Network.swift|The network implementation from MYONN book. Also includes Metal compute shader code.|
|NetworkView.swift|MVVM to make use of network in SwiftUI.|
|CanvasView.swift|The sketch area view for handwritten input.|
|MYONN-B30-0300-001-01.nnxd|Model in Neural Network Exchange Document (nndx) format. First functioning training. Mini-batch (size 30), learning rate 0.3, 1 epoch. Testset performance 0.9073.|
|MYONN-B30-0300-001-02.nnxd|Model in Neural Network Exchange Document (nndx) format. First functioning training with Swift concurrency. Mini-batch (size 30), learning rate 0.3, 1 epoch. Testset performance 0.8942.|
|MYONN-SGD-0300-001-01.nnxd|Model in Neural Network Exchange Document (nndx) format. First functioning training. Stoachstic gradient descent, learning rate 0.3, 1 epoch. Testset performance 0.9444.|
|MYONN-SGD-0300-001-02.nnxd|Model in Neural Network Exchange Document (nndx) format. First functioning training with Swift concurrency. Stoachstic gradient descent, learning rate 0.3, 1 epoch. Testset performance 0.9424.|

### References
- [Code Your Own Neural Network](https://www.amazon.de/Code-Neural-Network-step-step-ebook/dp/B00TXPGEHG) by Stephen C. Shaffer. A short but nevertheless valuable breviary with focus on practice.
- [Make Your Own Neural Network](https://www.amazon.de/-/en/Tariq-Rashid/dp/1530826608) by Tariq Rashid. A little more verbose than Shaffer with detailed step-by-step explanations yielding a network that recognizes handwritten digits.
- [The Architecture of Mind](http://faculty.otterbein.edu/dstucki/INST4200/Rumelhart.pdf) by David E. Rumelhart. Paper on back-propagation written by the inventor. Worth a reading.
