import cv2
import numpy
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image, ImageTk
import matplotlib
import colorsys


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def get_colors(image, number_of_colors):
	modified_image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

	clf = KMeans(n_clusters = number_of_colors)
	labels = clf.fit_predict(modified_image)

	counts = Counter(labels)

	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]
	rgb = numpy.array(rgb_colors, int)
	hsv = matplotlib.colors.rgb_to_hsv(rgb/255)
	hsv = numpy.column_stack((hsv[:,0]*360, hsv[:,1]*255, hsv[:,2]*255))
	hsv = numpy.round(hsv, decimals=0)
	hsv = hsv.astype(int)
	return hsv

def merge_horizontal(file1, file2, file3):
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)

    (width1, height1) = image1.size
    (width2, height2) = image2.size
    (width3, height3) = image3.size

    result_width = width1 + width2 + width3
    result_height = max(height1, height2, height3)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    result.paste(im=image3, box=(width1+width2, 0))
    result.save('output.jpg')
    return result
def merge_vertical(file1, file2, file3):
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)

    (width1, height1) = image1.size
    (width2, height2) = image2.size
    (width3, height3) = image3.size

    result_height = height1 + height2 + height3
    result_width = max(width1, width2, width3)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    result.paste(im=image3, box=(0, height1+height2))
    result.save('output.jpg')
    return result
def mergeMatrix(up_left,up_mid,up_right,mid_left,mid_mid,mid_right,bot_left,bot_mid,bot_right):
	row1 = merge_horizontal(up_left, mid_left, bot_left)
	row1.save('Output\\row1.jpg')
	row2 = merge_horizontal(up_mid, mid_mid, bot_mid)
	row2.save('Output\\row2.jpg')
	row3 = merge_horizontal(up_right, mid_right, bot_right)
	row3.save('Output\\row3.jpg')
	result = merge_vertical('Output\\row1.jpg', 'Output\\row2.jpg', 'Output\\row3.jpg')
	result.save('Output\\FullScore.jpg')
	return result

def getMusic(matrix, debug):
	music_array = numpy.empty(9, dtype='S32')
	if(debug):
		i = 0
		for x in matrix:
			if (x[1] < 1):
				if (0 < x[2] <= 25):
					music_array[i] = 'Sheetmusic\\dronea.jpg'
				if (25 < x[2] <= 50):
					music_array[i] = 'Sheetmusic\\droneb.jpg'
				if (50 < x[2] <= 75):
					music_array[i] = 'Sheetmusic\\dronec.jpg'
				if (75 < x[2] <= 100):
					music_array[i] = 'Sheetmusic\\droned.jpg'
			else:
				if (0 < x[0] <= 15 or 350 < x[0] <= 360):
					music_array[i] = 'Sheetmusic\\RED.jpg'
				if (15 < x[0] <= 45):
					music_array[i] = 'Sheetmusic\\ORANGE.jpg'
				if (45 < x[0] <= 70):
					music_array[i] = 'Sheetmusic\\YELLOW.jpg'
				if (70 < x[0] <= 90):
					music_array[i] = 'Sheetmusic\\PEA.jpg'
				if (90 < x[0] <= 155):
					music_array[i] = 'Sheetmusic\\GREEN.jpg'
				if (155 < x[0] <= 195):
					music_array[i] = 'Sheetmusic\\CYAN.jpg'
				if (195 < x[0] <= 240):
					music_array[i] = 'Sheetmusic\\BLUE.jpg'
				if (240 < x[0] <= 270):
					music_array[i] = 'Sheetmusic\\VIOLET.jpg'
				if (270 < x[0] <= 290):
					music_array[i] = 'Sheetmusic\\PURPLE.jpg'
				if (290 < x[0] <= 350):
					music_array[i] = 'Sheetmusic\\PINK.jpg'
			i = i+1
	else:
		i = 0
		for x in matrix:
			if (x[1] < 1):
				if (0 < x[2] <= 20):
					music_array[i] = 'Sheetmusic\\drone1_music.png'
				if (20 < x[2] <= 40):
					music_array[i] = 'Sheetmusic\\drone2_music.png'
				if (40 < x[2] <= 60):
					music_array[i] = 'Sheetmusic\\drone3_music.png'
				if (60 < x[2] <= 80):
					music_array[i] = 'Sheetmusic\\drone4_music.png'
				if (80 < x[2] <= 100):
					music_array[i] = 'Sheetmusic\\drone5_music.png'
			else:
				if (0 < x[0] <= 15 or 350 < x[0] <= 360):
					music_array[i] = 'Sheetmusic\\RED_music.png'
				if (15 < x[0] <= 45):
					music_array[i] = 'Sheetmusic\\ORANGE_music.png'
				if (45 < x[0] <= 70):
					music_array[i] = 'Sheetmusic\\YELLOW_music.png'
				if (70 < x[0] <= 90):
					music_array[i] = 'Sheetmusic\\PEA_music.png'
				if (90 < x[0] <= 155):
					music_array[i] = 'Sheetmusic\\GREEN_music.png'
				if (155 < x[0] <= 195):
					music_array[i] = 'Sheetmusic\\CYAN_music.png'
				if (195 < x[0] <= 240):
					music_array[i] = 'Sheetmusic\\BLUE_music.png'
				if (240 < x[0] <= 270):
					music_array[i] = 'Sheetmusic\\VIOLET_music.png'
				if (270 < x[0] <= 290):
					music_array[i] = 'Sheetmusic\\PURPLE_music.png'
				if (290 < x[0] <= 350):
					music_array[i] = 'Sheetmusic\\PINK_music.png'
			i = i+1

	up_left = music_array[0]
	up_mid = music_array[1]
	up_right = music_array[2]
	mid_left = music_array[3]
	mid_mid = music_array[4]
	mid_right = music_array[5]
	bot_left = music_array[6]
	bot_mid = music_array[7]
	bot_right = music_array[8]
	mergeMatrix(up_left,up_mid,up_right,mid_left,mid_mid,mid_right,bot_left,bot_mid,bot_right)
	return




#####################################################################
# HSV                                                               #
# Greyscale occurs when S = 0                                       #
#  - In this case, H controls nothing, and V controls brightness    #
#  - Create a divide to round greyscale-adjacent entries            #
# 	  - If S < 10, S = 0                                            #
#     - Top 2 rows                                                  #
#  - Sat. or more like brightness, and val. is more like lightness  #
#####################################################################

colors = numpy.array(get_colors(get_image('httyd\\test.jpg'), 9))
print(colors.reshape(3,3,3))

getMusic(colors, True)