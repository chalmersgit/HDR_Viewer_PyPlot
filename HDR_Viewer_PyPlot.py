'''
Andrew Chalmers, intitially written in 2019.

MIT License

Copyright (c) 2019 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import sys,os
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
import cv2

import importlib
import sphericalHarmonics as _sph
importlib.reload(_sph)

def linear2sRGB(hdr_img, gamma=2.2, autoExposure = 1.0):
	# Autoexposure
	hdr_img = hdr_img*autoExposure

	# Brackets
	lower = hdr_img <= 0.0031308
	upper = hdr_img > 0.0031308

	# Gamma correction
	hdr_img[lower] *= 12.92
	hdr_img[upper] = 1.055 * np.power(hdr_img[upper], 1.0/gamma) - 0.055

	# HDR to LDR format
	img_8bit = np.clip(hdr_img*255, 0, 255).astype('uint8')
	return img_8bit

def poleScale(y, width, relative=True):
	"""
	y = y pixel position (cast as a float)
	Scaling pixels lower toward the poles
	Sample scaling in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
	height = int(width/2)
	piHalf = np.pi/2
	pi4 = np.pi*4
	pi2OverWidth = (np.pi*2)/width
	piOverHeight = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	scaleFactor = (1.0 / pi4) * pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))
	if relative:
		scaleFactor /= (1.0 / pi4) * pi2OverWidth * (np.cos(piHalf - (piOverHeight / 2.0)) - np.cos(piHalf + (piOverHeight / 2.0)))
	return scaleFactor

def getPoleScaleMap(width, probability=False):
	height = int(width/2)
	#return np.repeat(poleScale(np.arange(0,height), width)[:, np.newaxis], width, axis=1)
	img = np.repeat(np.repeat(poleScale(np.arange(0,height), width)[:, np.newaxis], width, axis=1)[:,:,np.newaxis],3,axis=2)
	if probability:
		img /= np.sum(img)
	return img

def getSolidAngle(y, width, is3D=False):
	"""
	y = y pixel position (cast as a float)
	Solid angles in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
	height = int(width/2)
	pi2OverWidth = (np.pi*2)/width
	piOverHeight = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	return pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))

def getSolidAngleMap(width, probability=False):
	height = int(width/2)
	#return np.repeat(getSolidAngle(np.arange(0,height), width)[:, np.newaxis], width, axis=1)
	img = np.repeat(np.repeat(getSolidAngle(np.arange(0,height), width)[:, np.newaxis], width, axis=1)[:,:,np.newaxis],3,axis=2)
	if probability:
		img /= np.sum(img)
	return img

'''
def spherical_mean_and_std(img):
	weights = getSolidAngleMap(img.shape[1], probability=True)
	mean = np.sum(img*weights)
	std = np.sqrt(np.sum(((img-mean)**2)*weights))
	return mean, std
'''

def spherical_mean_and_std(values):
	if int(values.shape[1]/2) == values.shape[0] or int(values.shape[1]/4) == values.shape[0]:
		weights = getPoleScaleMap(values.shape[1])
		weights = weights[:values.shape[0],:,:] # incase the image is cropped (e.g bottom hemisphere removed)
		average = np.average(values, weights=weights)
		variance = np.average((values-average)**2, weights=weights)
		return (average, np.sqrt(variance))
	else:
		return (np.mean(values), np.std(values))

def getContrast(img):
	threshold,_ = spherical_mean_and_std(img)
	#threshold = np.median(img)
	return (np.max(img)-threshold)/threshold

def encode(x, a=5.0, b=2.2):
	x[x<0] = 0
	return (1.0/a)*x**(1.0/b)

def decode(x, a=5.0, b=2.2):
	return (a*x)**b

def landisOperator(x_src, expandDynRange=1.0, dynRangeExponent=2.0, threshold=None):
	if expandDynRange < 1:
		return x_src
	def remap(v):
		return (v-threshold)/(1.0-threshold)
	def mix(A,B,u):
		return A*(1.0-u)+B*u
	x = np.copy(x_src)
	x[x<0.0] = 0.0
	if threshold is None:
		threshold,_ = spherical_mean_and_std(x)
	#print('Threshold:',threshold)
	luminance = np.mean(x,axis=2)
	#expand_set = x[luminance>threshold,:]
	luminance_remapped = remap(luminance)
	dynMix = np.power(luminance_remapped,dynRangeExponent)
	for c in range(0,3):
		#x[luminance>threshold,c] = x[luminance>threshold,c]
		#x[luminance>threshold,c] = x[luminance>threshold,c]*expandDynRange
		#x[luminance>threshold,c] = dynMix[luminance>threshold]
		#x[luminance>threshold,c] = mix(x[luminance>threshold,c], np.power(x[luminance>threshold,c]+1.0,expandDynRange), dynMix[luminance>threshold])
		x[luminance>threshold,c] = mix(x[luminance>threshold,c], x[luminance>threshold,c]*expandDynRange, dynMix[luminance>threshold])
	return x

def getImgStats(img):
	minPxl = np.min(img)
	maxPxl = np.max(img)
	mean,std = spherical_mean_and_std(img)
	return minPxl, maxPxl, mean, std

def displayImage(img, windowWidth=8, resizeWidth=512):
	displayString = ""
	shiftDown = False
	showStats = True
	gamma = 2.2 # 1.0
	scaleFactor = 1.0
	contrast = getContrast(img)

	numrows, numcols, numdims = img.shape

	#print('Source shape:',img.shape)
	img = cv2.resize(img, (resizeWidth, int(resizeWidth/(img.shape[1]/img.shape[0]))), interpolation=cv2.INTER_AREA)
	#print('Display shape', img.shape)
	
	img_edited = np.copy(img)
	img_diffuse = None
	useDiffuse = False

	aspectRatio = float(img.shape[1])/img.shape[0]
	figWidth = windowWidth
	figHeight = int(figWidth/aspectRatio)

	fig = plt.figure(num='HDR Viewer', figsize=(figWidth,figHeight), frameon=False)
	ax = fig.add_axes([0, 0, 1, 1])
	limg = plt.imshow(np.clip(img*scaleFactor*255,0,255).astype(np.uint8))
	ax.axis('off')

	ui_text = ax.text(resizeWidth*0.01, resizeWidth*0.01, displayString, size=10, rotation=0.,
				 ha="left", va="top",
				 bbox=dict(boxstyle="round",
						   ec=(0.5, 0.5, 0.5, 0.5),
						   fc=(1., 1.0, 1.0, 0.5),))

	minPxl, maxPxl, mean, std = getImgStats(img)

	def format_coord(x, y):
		nonlocal gamma
		nonlocal scaleFactor
		nonlocal contrast
		nonlocal img_edited
		nonlocal minPxl, maxPxl, mean, std
		nonlocal displayString
		if not showStats:
			return ""
		if x is None and y is None:
			return ""

		col = int(x+0.5)
		row = int(y+0.5)
		if col>=0 and col<numcols and row>=0 and row<numrows:
			R,G,B = img_edited[row,col,:]
			M = (R+G+B)/3
			L = np.maximum(np.maximum(R,G),B)
			displayString =  'Hide stats (h)\nMove stats (right click)\nGamma (g):\t\t\t%1.1f\nExposure (scroll):\t%1.4f\n\nMouse =\t  (%d,%d)\nR =  \t\t\t%1.4f\nG =  \t\t\t%1.4f\nB =  \t\t\t%1.4f\nAve =   \t\t%1.4f\nLargest = \t%1.4f\n\nGlobal:\nResolution: %dx%d\nmin:\t\t\t%1.4f\nmax:   \t\t%1.4f\nmean: \t\t%1.4f\nstd: \t\t\t%1.4f\ncontrast: \t%1.4f\n' %(gamma, scaleFactor, col,row,R,G,B,M,L,numcols,numrows, minPxl,maxPxl, mean,std, contrast)
			#displayString = displayString.expandtabs()
			displayString = displayString.replace("\t", "    ")
			ui_text.set_text(str(displayString))
			refreshCanvas()

			'''
			displayString = 	'Resolution: %dx%d, Mouse: (%d,%d)\n\
						R=%1.4f, G=%1.4f, B=%1.4f, Ave=%1.4f, Largest=%1.4f\n\
						Gamma (g):%1.1f\n\
						Exposure (scroll):%1.4f\n\
						min=%1.4f, max=%1.4f\n\
						mean=%1.4f, std=%1.4f\n\
						contrast=%1.4f\n' %(numcols,numrows,col,row, R,G,B,M,L, gamma, scaleFactor, minPxl,maxPxl, mean,std, contrast)
			'''
			#return displayString
			return ""
		else:
			#return 'x=%1.4f, y=%1.4f'%(x, y)
			return ""

	ax.format_coord = format_coord
	
	#print(format_coord)
	#ax.text(resizeWidth//2, resizeWidth//4, format_coord, size=10, rotation=0.,
	#			 ha="center", va="center",
	#			 bbox=dict(boxstyle="round",
	#					   ec=(1., 0.5, 0.5),
	#					   fc=(1., 0.8, 0.8),))

	def refreshCanvas():
		fig.canvas.draw_idle()


	def draw():
		nonlocal img
		nonlocal scaleFactor
		nonlocal gamma
		nonlocal contrast
		nonlocal img_edited
		nonlocal img_diffuse
		nonlocal useDiffuse
		nonlocal minPxl, maxPxl, mean, std
		nonlocal displayString

		#img_edited = landisOperator(img, expandDynRange=scaleFactor)
		contrast = getContrast(img_edited)
		minPxl, maxPxl, mean, std = getImgStats(img_edited)

		#limg.set_data(np.clip(img*255,0,255).astype(np.uint8))
		#limg.set_data(np.clip(img_edited*255,0,255).astype(np.uint8))
		#limg.set_data(np.clip(pow(img_edited,1.0/gamma)*255,0,255).astype(np.uint8))
		#limg.set_data(np.clip(pow(img_edited,1.0/gamma)*scaleFactor*255,0,255).astype(np.uint8))
		
		if useDiffuse:
			limg.set_data(linear2sRGB(img_diffuse, gamma=gamma, autoExposure=scaleFactor))
		else:
			limg.set_data(linear2sRGB(img_edited, gamma=gamma, autoExposure=scaleFactor))

		#ui_text.set_position((x[tpos], f[tpos]))
		#ui_text.set_text(str(displayString))

		refreshCanvas()

		return img.astype(np.float32)

	def on_scroll(event):
		nonlocal scaleFactor
		nonlocal shiftDown
		scaleUp_fast = 3.0
		scaleUp_slow = 1.2
		if event.button=='up':
			if shiftDown:
				scaleFactor *= scaleUp_fast
			else:
				scaleFactor *= scaleUp_slow
			#scaleFactor +=0.1
		else:
			#scaleFactor -=0.1
			if shiftDown:
				scaleFactor *= (1/scaleUp_fast)
			else:
				scaleFactor *= (1/scaleUp_slow)
			if scaleFactor<0:
				scaleFactor = 0.00001
		# TODO update UI exposure value
		#ui_text.set_text(str(displayString))
		#refreshCanvas()
		if event.xdata is not None and event.ydata is not None:
			format_coord(event.xdata, event.ydata)
		draw()

	def on_key_press(event):
		nonlocal shiftDown
		if event.key == 'shift':
			shiftDown = True

	def on_key_release(event):
		nonlocal gamma
		nonlocal scaleFactor
		nonlocal img_edited
		nonlocal img_diffuse
		nonlocal useDiffuse
		nonlocal showStats
		nonlocal shiftDown
		if event.key == 'shift':
			shiftDown = False
		if event.key == 'g':
			if gamma > 1.01:
				gamma = 1.0
			else:
				gamma = 2.2
			format_coord(event.xdata, event.ydata)
			draw()
		if event.key == 'd':
			if img_diffuse is None:
				img_diffuse = _sph.getDiffuseMapFromImg(img_edited, width=img_edited.shape[1], widthLowRes=32)
			useDiffuse = not useDiffuse 
			draw()
		if event.key == 'r':
			gamma = 2.2
			scaleFactor = 1.0
			format_coord(event.xdata, event.ydata)
			draw()
		if event.key == 'h':
			showStats = not showStats
			if not showStats:
				ui_text.set_text("")
				refreshCanvas()
		if event.key == 'o':
			path, file = os.path.split(os.path.realpath(__file__))
			path = os.path.join(path, 'output')
			print(path)
			if not os.path.exists(path):
				print("Making:",path)
				os.makedirs(path)
			outputPathEXR = os.path.join(path, '_output.exr')
			outputPathHDR = os.path.join(path, '_output.hdr')
			outputPathJPG = os.path.join(path, '_output.jpg')
			print("Saving to",outputPathEXR)
			print("Saving to",outputPathHDR)
			print("Saving to",outputPathJPG)
			im.imwrite(outputPathEXR, img_edited)
			im.imwrite(outputPathHDR, img_edited)
			im.imwrite(outputPathJPG, linear2sRGB(img_edited, gamma=gamma, autoExposure=scaleFactor))

	def on_click(event):
		if event.button==3: # right click
			ui_text.set_position((event.xdata, event.ydata))
			refreshCanvas()

	def on_motion(event):
		if event.button==3 and event.xdata is not None and event.ydata is not None: # right click
			ui_text.set_position((event.xdata, event.ydata))
			refreshCanvas()

	fig.canvas.mpl_connect('key_press_event', on_key_press)
	fig.canvas.mpl_connect('key_release_event', on_key_release)
	fig.canvas.mpl_connect('scroll_event', on_scroll)
	fig.canvas.mpl_connect('button_press_event', on_click)
	fig.canvas.mpl_connect('motion_notify_event', on_motion)

	draw()
	#fig.tight_layout(rect=[0,0.1,1,1]) 
	plt.show()

if __name__ == "__main__":
	print('HDR Viewer')
	#sys.argv = ['', './images/grace-new.exr']
	#sys.argv = ['', './images/grace-new.exr', 512]
	if len(sys.argv)<2:
		print("Missing args: [string filepath] optional: [int resizeWidth]")
		sys.exit()

	fn = sys.argv[1]

	img = im.imread(fn)

	if len(img.shape)==2:
		img = np.stack((img,)*3, axis=-1)
	else:
		img = img[:,:,:3]


	resizeWidth = 512
	try:
		if len(sys.argv)>2:
			resizeWidth = int(sys.argv[2])
	except:
		pass

	#img = img/np.max(img)

	displayImage(img, resizeWidth=resizeWidth)

	print('Exit')
