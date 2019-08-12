from array import array
from cluster import distance_euclidean as distance
import argparse

########################################################################
#                              Task 5                                  #
########################################################################


def read_image(filename):
	'''
	File format: PPM image or PGM image

	Returns image as a 3D list of size [h,w,3] if rgb image and 2D list of size [h,w] if grayscale image

	[Empty RGB image as 3D list] -- img = [[[0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
	[Empty grayscale image as 2D list] -- img = [[0 for _ in range(width)] for _ in range(height)]
	'''
	img = None
	# todo: task5
	# Here you need to read the file, and convert the data into a 3D or 2D list depending on the image type.
	# If first 2 bytes of the file are 'P6', it's an RGB image and if first 2 bytes are 'P5', it's a grayscale image.
	# After that, there are 2 numbers written as strings which correspond to the width and height of the image
	# Then there is one more number written as string which corresponds to the max. value of a pixel in the image (not really required here)
	# Afterwards, all data in the file corresponds to the actual image.
	# You can read the data byte by byte and update corresponding element of 'img' variable.
	# For more information on PPM and PGM file formats, refer to the link mentioned in the assignment document.

	f = open(filename, "rb")
	if (f.read(2) == b'P6'):
		# print("color")
		f.read(1)
		c = 0
		temp = f.read(1)
		while(temp != b' '):
			c = c*10 + int(temp)
			temp = f.read(1)
		w = c
		# print(w)
		c = 0
		temp = f.read(1)
		while(temp != b'\n'):
			c = c*10 + int(temp)
			temp = f.read(1)
		h = c
		# print(h)

		img = [[[0 for _ in range(3)] for _ in range(w)] for _ in range(h)]

		while(f.read(1) != b'\n'):
			blablab = 1
		i = 0
		j = 0
		cl = 0
		it = h*w*3
		while (it):
			num = int(str(f.read(1)).encode("hex"),16)
			img[i][j][cl] = num
			cl = cl + 1
			if (cl == 3):
				j = j + 1
				if (j == w):
					j = 0 
					i = i + 1
				cl = 0
			it  = it - 1
	else:
		# print("grey")
		f.read(1)
		c = 0
		temp = f.read(1)
		while(temp != b' '):
			c = c*10 + int(temp)
			temp = f.read(1)
		w = c
		# print(w)
		c = 0
		temp = f.read(1)
		while(temp != b'\n'):
			c = c*10 + int(temp)
			temp = f.read(1)
		h = c
		# print(h)
		img = [[0 for _ in range(w)] for _ in range(h)]
		# print(img)
		while(f.read(1) != b'\n'):
			blalba = 1
		i = 0
		j = 0
		it = h*w
		while (it):
			num = int(str(f.read(1)).encode("hex"),16)
			# if (it == h*w):
			# 	print(num)
			img[i][j] = num
			# print("test")
			j = j + 1
			if (j == w):
				j = 0 
				i = i + 1
			it  = it - 1
		# print(img)

	print(len(img))
	return img


def preprocess_image(img):
	'''
	Transforms image into format suitable for clustering.

	Returns data - a list of datapoints where each datapoint is a tuple
	'''
	data = [(None,None,None)]
	# todo: task5
	# You need to convert the image such that data is a list of datapoints on which you want to do clustering and each datapoint in the list is a 3-tuple with RGB values.

	h = len(img)
	w = len(img[0])
	# if (isinstance(img[0][0], list)):
	data =[tuple(img[i][j]) for i in range(h) for j in range(w)]
	return data


########################################################################
#                              Task 6                                  #
########################################################################


def label_image(img, cluster_centroids):
	'''
	img: RGB image as 3D list of size [h,w,3]

	Return a 2D matrix where each point stores the nearest cluster centroid of corresponding pixel in the image
	[Empty 2D list] -- labels = [[0 for _ in range(width)] for _ in range(height)]
	'''

	h = len(img)
	w = len(img[0])
	cluster_labels = [[0 for _ in range(w)] for _ in range(h)]
	for i in range(h):
		for j in range(w):
			p = tuple(img[i][j])
			mind = float("inf")
			mini = 0
			for k in range(len(cluster_centroids)):
				dist = distance(p, tuple(cluster_centroids[k]))
				if (dist < mind):
					mind  = dist
					mini = k
			cluster_labels[i][j] = mini

	# todo: task6
	# Iterate over the image pixels and replace each pixel value with cluster number corresponding to its nearest cluster centroid

	return cluster_labels


def write_image(filename, img):
	'''
	img: 3D list of size [h,w,3] if rgb image and 2D list of size [h,w] if grayscale image

	File format: PPM image if img is rgb and PGM if image is grayscale
	'''
	# todo: task6
	# Create a new file with the given file name (assume filename already includes extension).
	# First line in the file should contain the format code - P5 for grayscale (PGM) and P6 for RGB (PPM)
	# Next line should contain the width and height of the image in format - width height
	# Next line should contain the maximum value of any pixel in the image (use max function to find it). Note that this should be an integer and within (0-255)
	# Next line onwards should contain the actual image content as per the binary PPM or PGM file format.
	f = open(filename, "wb")
	h = len(img)
	w = len(img[0])
	if(isinstance(img[0][0], list)):
		f.write(b"P6")
		f.write(b"\n")
		maxi = img[0][0][0]
		for i in range(h):
			for j in range(w):
				for k in range(3):
					if (img[i][j][k] > maxi):
						maxi = img[i][j][k]

		f.write(str(w) + " " + str(h) + "\n" + str(maxi) + "\n")
		for i in range(h):
			for j in range(w):
				for k in range(3):
					# print("color")
					f.write(bytes(bytearray([img[i][j][k]])))
	else:
		f.write(b"P5")
		f.write(b"\n")
		maxi = img[0][0]
		for i in range(h):
			for j in range(w):
				if (img[i][j] > maxi):
					maxi = img[i][j]

		f.write(str(w) + " " + str(h) + "\n" + str(maxi) + "\n")
		
		for i in range(h):
			for j in range(w):
				# print("grey")
				f.write(bytes(bytearray([img[i][j]])))

########################################################################
#                              Task 7                                  #
########################################################################


def decompress_image(cluster_labels, cluster_centroids):
	'''
	cluster_labels: 2D list where each point stores the nearest cluster centroid of corresponding pixel in the image
	cluster_centroids: cluster centroids

	Return 3D list of size [h,w,3] if rgb image and 2D list of size [h,w] if grayscale image
	[Empty RGB image as 3D list] -- img = [[[0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
	'''

	# img = None
	# todo: task7
	# Iterate over the 2D list's elements and replace each value (cluster label) with value of its corresponding cluster centroid
	# Use distance function (using distance from cluster.py to find the nearest cluster)
	h = len(cluster_labels)
	w = len(cluster_labels[0])
	if(len(cluster_centroids[0]) == 3):
		# print("color")
		img = [[[0 for _ in range(3)] for _ in range(w)] for _ in range(h)]
		for i in range(h):
			for j in range(w):
				for k in range(3):
					img[i][j][k] = int(cluster_centroids[cluster_labels[i][j]][k])
	else:
		# print("grey")
		img = [[0 for _ in range(w)] for _ in range(h)]
		for i in range(h):
			for j in range(w):
				img[i][j] = int(cluster_centroids[cluster_labels[i][j]])
	return img


########################################################################
#                       DO NOT EDIT THE FOLLWOING                      #
########################################################################


def readfile(filename):
	'''
	File format: Each line contains a comma separated list of real numbers, representing a single point.
	Returns a list of N points, where each point is a d-tuple.
	'''
	data = []
	with open(filename, 'r') as f:
		data = f.readlines()
	data = [tuple(map(float, line.split(','))) for line in data]
	return data


def writefile(filename, points):
	'''
	points: list of tuples
	Writes the points, one per line, into the file.
	'''
	if filename is None:
		return
	with open(filename, 'w') as f:
		for m in points:
			f.write(','.join(map(str, m)) + '\n')
	print 'Written points to file ' + filename


def parse():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')

	preprocess_parser = subparsers.add_parser('preprocess')
	preprocess_parser.add_argument(dest='image', type=str, help='Image filename (ppm)')
	preprocess_parser.add_argument('-o', '--output', dest='output', default='image_data.csv', type=str, help='Output Data filename (csv). Default: image_data.csv')

	compress_parser = subparsers.add_parser('compress')
	compress_parser.add_argument(dest='image', type=str, help='Image filename (ppm)')
	compress_parser.add_argument(dest='centroids', type=str, help='Cluster centroids (csv)')
	compress_parser.add_argument('-o', '--output', dest='output', default='labels.pgm', type=str, help='Output Image filename (pgm). Default: labels.pgm')

	decompress_parser = subparsers.add_parser('decompress')
	decompress_parser.add_argument(dest='labels', type=str, help='Labels filename (pgm)')
	decompress_parser.add_argument(dest='centroids', type=str, help='Cluster centroids (csv)')
	decompress_parser.add_argument('-o', '--output', dest='output', default='decompressed.ppm', type=str, help='Output Image filename (ppm). Default: decompressed.ppm')

	return parser.parse_args()


if __name__ == '__main__':

	args = parse()
	if args.command == 'preprocess':
		img = read_image(args.image)
		data = preprocess_image(img)
		writefile(args.output, data)
	elif args.command == 'compress':
		img = read_image(args.image)
		centroids = readfile(args.centroids)
		cluster_labels = label_image(img, centroids)
		write_image(args.output, cluster_labels)
	else:
		cluster_labels = read_image(args.labels)
		centroids = readfile(args.centroids)
		img = decompress_image(cluster_labels, centroids)
		write_image(args.output, img)
