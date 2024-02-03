# AIplay
Das Repository enthält die Sourcen der App zum Developer's Corner Artikel _Das eigene KI-Modell programmieren_ im Apple-Magazin [Mac & i Heft 1/2024, S. 126](https://www.heise.de/select/mac-and-i/2024/1/2326112085400864712).

A SwiftUI implementation of the textbook Make Your Own Neural Network (MYONN). The app downloads the MNIST dataset and enables training and prediction using the files included in the set. A sketch area accepts handwritten input for prediction. There are save and load functions for trained models in a simple and convenient format.

The app features stochastic gradient descent (SGD) and mini-batch which is the default. The default network configuration is 784/ 100/ 10 fully-connected input/ hidden/ output nodes with sigmoid activation. To enable SGD set Mini-batch size in setup to 1.



### Usage
The app is on [TestFlight](https://testflight.apple.com/join/M2uSLM1e). After installation it asks on the first start for a folder to store the MNIST dataset. The four MNIST status indicators turn into green when all files have been downloaded and unzipped. There is some basic usage information when tapping the circled i in the top right corner.

Input from the sketch area must be centered with some padding to the edges. Ideally, the images in the _Query Input_ and _Query Image_ areas will look fairly similar in terms of their position in their respective areas.

**Working**
- MNIST loading/ training/ predicting
- Save/ load trained model (proprietary format)
- Predict handwritten input
- Simple configuration UI

**Wanted**
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
|Matrix.swift|A somewhat pimped Array type for matrix operations. Leverages vDSP.|
|MNISTView.swift|MVVM to put MNIST on disk.|
|Network.swift|The network implementation from MYONN book. Also includes Metal compute shader code.|
|NetworkView.swift|MVVM to make use of network in SwiftUI.|
|CanvasView.swift|The sketch area view for handwritten input.|
|SetupView.swift|Simple network and layer configuration UI.|
|default-model.nnxd|Model in Neural Network Exchange Document (nnxd) format. Loaded on first app launch. Mini-batch (size 30), learning rate 0.3, 7 epochs. Testset performance 0.9524.|

### References
- [Code Your Own Neural Network](https://www.amazon.de/Code-Neural-Network-step-step-ebook/dp/B00TXPGEHG) by Stephen C. Shaffer. A short but nevertheless valuable breviary with focus on practice.
- [Make Your Own Neural Network](https://www.amazon.de/-/en/Tariq-Rashid/dp/1530826608) by Tariq Rashid. A little more verbose than Shaffer with detailed step-by-step explanations yielding a network that recognizes handwritten digits.
- [The Architecture of Mind](http://faculty.otterbein.edu/dstucki/INST4200/Rumelhart.pdf) by David E. Rumelhart. Paper on back-propagation written by the inventor. Worth a reading.
