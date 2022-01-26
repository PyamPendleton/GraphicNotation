import cv2
import numpy
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image


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
	
	return rgb

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

def getMusic(rgb_matrix):
	

##################
##################

# Warm -> Cool = Articulations
# Cool colors = high, airy texture ()
# Warm colors = low, sonorous texture ()
# White -> Black = Drones
# Drone players keep time
# Bright colors = sparse texture
# Dark colors = dense texture

# Lighter = equal, large values
# Darker = equal, low values


colors = numpy.array(get_colors(get_image('Images\\httyd.jpg'), 9))
print(colors.reshape(3,3,3))

mergeMatrix('Images\\Figure_1.png', 'Images\\Figure_2.png', 'Images\\test.jpg', 
	'Images\\Figure_1.png', 'Images\\Figure_2.png', 'Images\\test.jpg', 
	'Images\\Figure_1.png', 'Images\\Figure_2.png', 'Images\\test.jpg')