#!/usr/bin/python

import jetson.inference
import jetson.utils

import argparse

#parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename",type=str,help="filename of image to process")
parser.add_argument("--network",type=str,default="googlenet",help="model to use, can be: googlenet, resnet-18, etc. (see --help for others)")
opt = parser.parse_args()

# load an image into shared cpu/gpu memory
img, width, height = jetson.utils.loadImageRGBA(opt.filename)

#load recognition network
net = jetson.inference.imageNet(opt.network)

#classify image
class_idx, confidence = net.Classify(img, width, height)

# find object description
class_desc = net.GetClassDesc(class_idx)

# print the result
print("image is recognized as '{:s}' (class {:d}) with {:f}% confidence".format(class_desc, class_idx,confidence * 100))
