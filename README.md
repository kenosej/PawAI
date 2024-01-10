# PawAI

"PawAI" represents a comprehensive approach to image analysis, utilizing a combination of the Haar cascade classifier for face detection, the VGG-16 model for recognizing dogs, and PawAI for identifying dog breeds. In the first step, the Haar cascade classifier provides satisfactory results in facial detection, marking faces with rectangles in test images. Additionally, the code is extended to enable precise counting of detected faces in the image.

Subsequently, the VGG-16 model is employed for highly accurate dog recognition, and the `dog_detector` function analyzes the presence of dogs in the images. This step provides a deeper understanding of the model's performance in the specific context of test images, identifying the presence of dogs with a high degree of accuracy.

The latest addition, PawAI, enhances the analysis by enabling the recognition of dog breeds. Thanks to its ability to recognize a diverse range of breeds, PawAI offers a detailed insight into the unique characteristics associated with each breed. This extension adds depth and complexity to image analysis, facilitating the identification and classification of dogs not only as simple objects but also recognizing specific characteristics related to their breeds.

Overall, this code not only provides a fundamental analysis of the presence of objects of interest in images but also explores additional dimensions of face and dog recognition, including their breed characteristics, making it a versatile and comprehensive tool in the field of image analysis.

<hr>

**Installation**

1. Clone the repository with the latest commit
2. Download the datasets linked [here]()
3. Make sure you have Python v3.11 installed
4. Open the project in preffered editor
5. Run ```pip install -r requierments.txt``` from the root folder location
6. Run ```python.exe main.py```
   
