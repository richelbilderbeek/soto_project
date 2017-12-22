#!/usr/bin/env python

import numpy as np
import cv2
from time import clock
from numpy import pi, sin, cos
import common

source = "3f_1.mp4"
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if cap is None or not cap.isOpened():
  # print 'Warning: unable to open video source: ', source
  # if fallback is not None:
  #    return create_capture(fallback, None)
  print "Error: unable to open video source: '" + source + "'"
  raise SystemExit

print "Gelukt!"

