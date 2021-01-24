# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 22:04:35 2021

@author: Sooryaprakash
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local
from skimage.filters import *
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan', methods=['GET', 'POST'])
def upload_file():
    
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
      
      
      imgorg = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      ratio = imgorg.shape[0] / 500.0

      y = round(imgorg.shape[0] / ratio)
      x = round(imgorg.shape[1] / ratio)
      
      img = cv2.resize(imgorg,(x,y))

      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
      edged = cv2.Canny(gray, 75, 200)
        
      cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        
      for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screenCnt = approx
            break
        
      warped = four_point_transform(imgorg, screenCnt.reshape(4, 2) * ratio)
        
      warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
      T = threshold_local(warped,61, offset = 10, method = "gaussian")
      warped = (warped > T).astype("uint8") * 255
        
      cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'_scanned')+filename,warped)

      return redirect(url_for('static', filename='uploads/' + '_scanned'+filename), code=301)


def order_points_old(pts):

	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):

  rect = order_points_old(pts)
  (tl, tr, br, bl) = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

  return warped





		
        
if __name__ == "__main__":
    app.run(debug=True)

