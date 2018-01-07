# Precog_TaskB

## Test Cases

<p align="center">
  <img src="https://raw.githubusercontent.com/shantanujain/Precog_TaskB_2/master/samples/sample1.png" width="50%" >
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/shantanujain/Precog_TaskB_2/master/samples/sample2.png" width="50%" >
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/shantanujain/Precog_TaskB_2/master/samples/sample3.png" width="50%" >
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/shantanujain/Precog_TaskB_2/master/samples/sample4.png" width="50%" >
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/shantanujain/Precog_TaskB_2/master/samples/sample5.png" width="50%" >
</p>


## Steps

* Scraped images of Narendra Modi and Arvind Kejriwal from Google
* Extracted features of the face (mouth, eyebrow, eyes, nose, jaw) using dlib which can be compared with test images to predict Kejriwal/Modi

## Prerequisites

* Flask
* OpenCV
* dlib
* numpy

## To Run

```shell
python app.py
python detect.py *path-to-image*
```
