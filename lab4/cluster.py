from copy import deepcopy
from itertools import cycle
from pprint import pprint as pprint
import sys
import argparse
import matplotlib.pyplot as plt
import random
import math


########################################################################
#                              Task 1                                  #
########################################################################


def distance_euclidean(p1, p2):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point

	Returns the Euclidean distance b/w the two points.
	'''

	distance = 0
	for i,j in zip(p1, p2):
		distance = distance + (i-j)*(i-j)
	distance = math.sqrt(distance)
	# TODO [task1]:
	# Your function must work for all sized tuples.

	########################################
	return distance


def initialization_forgy(data, k):
	'''
	data: list of tuples: the list of data points
	k: int: the number of cluster centroids to return

	Returns a list of tuples, representing the cluster centroids
	'''
	
	centroids = random.sample(data, k)

	# TODO [task1]:
	# Initialize the cluster centroids by sampling k unique datapoints from data

	########################################
	assert len(centroids) == k
	return centroids


def kmeans_iteration_one(data, centroids, distance):
	'''
	data: list of tuples: the list of data points
	centroids: list of tuples: the current cluster centroids
	distance: callable: function implementing the distance metric to use

	Returns a list of tuples, representing the new cluster centroids after one iteration of k-means clustering algorithm.
	'''
	dim = len(data[0])
	sum_cents = [[0.0]*dim for _ in centroids]
	new_centroids = []
	c = [0.0 for _ in centroids]
	for i in range(len(data)):
		temp = [distance(data[i],j) for j in centroids]
		lab = temp.index(min(temp))
		c[lab] = c[lab] + 1.0
		sum_cents[lab] = [sum(x) for x in zip(sum_cents[lab],data[i])]
	for i in range(len(centroids)):
		if c[i] == 0:
			new_centroids.insert(i,centroids[i])
		else:
			new_centroids.insert(i,tuple([k/c[i] for k in sum_cents[i]]))

	# new_centroids = [[] for i in centroids]
	# # print(new_centroids)
	# for i in data:
	# 	mind = float("inf")
	# 	mini = 0
	# 	for j in range(len(centroids)):
	# 		dist = distance(i,centroids[j])
	# 		if (dist < mind):
	# 			mind = dist
	# 			mini = j
	# 	# print("test")
	# 	new_centroids[mini].insert(0, i)
	# # print("--------", new_centroids, "--------------")
	# newc = []
	# for i in range(len(new_centroids)):
		
	# 	if(len(new_centroids[i]) == 0):
	# 		newc.insert(i, centroids[i])
	# 	elif(len(new_centroids[i]) == 1):
	# 		newc.insert(i, new_centroids[i][0])
	# 	else :
	# 		temp = [0.0 for _ in range(len(data[0]))]
	# 		c = 0
	# 		for j in new_centroids[i]:
	# 			temp = [sum(x) for x in zip(temp,j)]
	# 		den = len(new_centroids[i])
	# 		temp = [k/(1.0*den) for k in temp]
	# 		newc.insert(i, tuple(temp))
	# TODO [task1]:
	# You must find the new cluster centroids.
	# Perform just 1 iteration (assignment+updation) of k-means algorithm.

	########################################
	assert len(new_centroids) == len(centroids)
	return new_centroids


def hasconverged(old_centroids, new_centroids, epsilon=1e-1):
	'''
	old_centroids: list of tuples: The cluster centroids found by the previous iteration
	new_centroids: list of tuples: The cluster centroids found by the current iteration

	Returns true iff no cluster centroid moved more than epsilon distance.
	'''

	converged = True
	diff = [distance_euclidean(old_centroids[i], new_centroids[i]) for i in range(len(old_centroids))]
	for i in diff:
		if (i > epsilon):
			converged = False
			break;

	# TODO [task1]:
	# Use Euclidean distance to measure centroid displacements.

	########################################
	return converged


def iteration_many(data, centroids, distance, maxiter, algorithm, epsilon=1e-1):
	'''
	maxiter: int: Number of iterations to perform

	Uses the iteration_one function.
	Performs maxiter iterations of the clustering algorithm, and saves the cluster centroids of all iterations.
	Stops if convergence is reached earlier.

	Returns:
	all_centroids: list of (list of tuples): Each element of all_centroids is a list of the cluster centroids found by that iteration.
	'''

	all_centroids = []
	all_centroids.append(centroids)
	# prevc = deepcopy(centroids)
	for _ in range(maxiter):
		prevc = deepcopy(all_centroids[-1])
		currc = deepcopy(iteration_one(data, prevc, distance, algorithm))
		all_centroids.append(currc)
		if(hasconverged(prevc, currc)):
			break;
		# prevc = deepcopy(currc)
		

	# TODO [task1]:
	# Perform iterations by calling the iteration_one function multiple times. Make sure to pass the algorithm argument to iteration_one (already defined).
	# Stop only if convergence is reached, or if max iterations have been exhausted.
	# Save the results of each iteration in all_centroids.
	# Tip: use deepcopy() if you run into weirdness.

	########################################
	return all_centroids


def performance_SSE(data, centroids, distance):
	'''
	data: list of tuples: the list of data points
	centroids: list of tuples: representing the cluster centroids

	Returns: The Sum Squared Error of the clustering represented by centroids, on the data.
	'''
	
	sse = 0.0
	# print("---------", centroids, "SSE-------------")
	c = 0
	for i in data:
		# print("--------", i, "--------------")
		mind = float("inf")
		# print(centroids)
		for j in centroids:
			# print("-", j, "-")
			if (distance(i,j) < mind):
				mind = distance(i,j)
		# print(mind)
		sse = sse + mind*mind

	# TODO [task1]:
	# Calculate the Sum Squared Error of the clustering represented by centroids, on the data.
	# Make sure to use the distance metric provided.

	########################################
	return sse


########################################################################
#                              Task 3                                  #
########################################################################

# def nSample(distribution, values, n):
#     if sum(distribution) != 1:
#         distribution = normalize(distribution)
#     rand = [random.random() for i in range(n)]
#     rand.sort()
#     samples = []
#     samplePos, distPos, cdf = 0,0, distribution[0]
#     while samplePos < n:
#         if rand[samplePos] < cdf:
#             samplePos += 1
#             samples.append(values[distPos])
#         else:
#             distPos += 1
#             cdf += distribution[distPos]
#     return samples

def one_Sample(dist, vals):
	rand = random.random()*sum(dist) #As dist(array of square of minimum distances) is not randomised i am multiplying random number with sum(dist)
	temp = dist[0]
	c = 0
	# checking the occurence of random number in cdf of min squared distances
	for c in range(len(dist)):
		if(rand <= temp):
			return vals[c]
		c = c + 1
		temp = temp + dist[c]

def initialization_kmeansplusplus(data, distance, k):
	'''
	data: list of tuples: the list of data points
	distance: callable: a function implementing the distance metric to use
	k: int: the number of cluster centroids to return

	Returns a list of tuples, representing the cluster centroids
	'''

	
	centroids = random.sample(data, 1) #first centroid
	for k1 in range(1, k):
		dist = [] # list of square minimum distances from every point to centroids
		for i in range(len(data)):
			dat = data[i]
			mind = float("inf")
			for j in centroids:
				dis = distance(dat, j)
				# print("test")
				if(dis < mind):
					mind  = dis
			dist.insert(i, mind*mind) # insert the ith data point squared minimum distance from dentroids
		centroids.insert(0, one_Sample(dist, data))
	# TODO [task3]:
	# Use the kmeans++ algorithm to initialize k cluster centroids.
	# Make sure you use the distance function given as parameter.

	# NOTE: Provide extensive comments with your code.

	########################################
	assert len(centroids) == k
	return centroids


########################################################################
#                              Task 4                                  #
########################################################################


def distance_manhattan(p1, p2):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point

	Returns the Manhattan distance b/w the two points.
	'''

	# default k-means uses the Euclidean distance.
	# Changing the distant metric leads to variants which can be more/less robust to outliers,
	# and have different cluster densities. Doing this however, can sometimes lead to divergence!

	distance = 0
	for i,j in zip(p1, p2):
		distance = distance + abs(i-j)
	
	# TODO [task4]:
	# Your function must work for all sized tuples.

	########################################
	return distance


def kmedians_iteration_one(data, centroids, distance):
	'''
	data: list of tuples: the list of data points
	centroids: list of tuples: the current cluster centroids
	distance: callable: function implementing the distance metric to use

	Returns a list of tuples, representing the new cluster centroids after one iteration of k-medians clustering algorithm.
	'''
	# dim = len(data)

	new_centroids = [[] for i in centroids]
	for i in data:
		mind = float("inf")
		mini = 0
		for j in range(len(centroids)):
			dist = distance(i,centroids[j])
			if (dist < mind):
				mind = dist
				mini = j
		new_centroids[mini].insert(0, i)
	# print("--------", new_centroids, "-----------------")
	newc = []
	for i in range(len(new_centroids)):
		if(len(new_centroids[i]) == 0):
			newc.insert(i, centroids[i])
		elif(len(new_centroids[i]) == 1):
			newc.insert(i, new_centroids[i][0])
		else:
			temp = []
			c = 0
			for j in zip(*(new_centroids[i])):
				# print("-----", j, "---------")
				d = deepcopy(j)
				# print("test")
				d = list(d)
				d.sort()
				# print(dim)
				dim = len(d)
				if (dim%2 == 1):
					temp.insert(c, d[(dim-1)/2])
				else:
					temp.insert(c, (d[(dim/2) - 1] + d[(dim/2)])/2.0)
				c = c+1
			# print(temp)
			newc.insert(i, tuple(temp))
		# print(newc)


	# TODO [task4]:
	# You must find the new cluster centroids.
	# Perform just 1 iteration (assignment+updation) of k-medians algorithm.

	########################################
	assert len(newc) == len(centroids)
	return newc


def performance_L1(data, centroids, distance):
	'''
	data: list of tuples: the list of data points
	centroids: list of tuples: representing the cluster centroids

	Returns: The L1-norm error of the clustering represented by centroids, on the data.
	'''

	sse = 0.0
	for i in data:
		mind = float("inf")
		for j in centroids:
			if (distance_manhattan(i,j) < mind):
				mind = distance_manhattan(i,j)
		sse = sse + mind

	# TODO [task4]:
	# Calculate the L1-norm error of the clustering represented by centroids, on the data.
	# Make sure to use the distance metric provided.

	########################################
	return sse


########################################################################
#                       DO NOT EDIT THE FOLLWOING                      #
########################################################################


def argmin(values):
	return min(enumerate(values), key=lambda x: x[1])[0]


def avg(values):
	return float(sum(values))/len(values)


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


def writefile(filename, centroids):
	'''
	centroids: list of tuples
	Writes the centroids, one per line, into the file.
	'''
	if filename is None:
		return
	with open(filename, 'w') as f:
		for m in centroids:
			f.write(','.join(map(str, m)) + '\n')
	print 'Written centroids to file ' + filename


def iteration_one(data, centroids, distance, algorithm):
	'''
	algorithm: algorithm to use {kmeans, kmedians}

	Uses the kmeans_iteration_one or kmedians_iteration_one function as required.

	Returns a list of tuples, representing the new cluster centroids after one iteration of clustering algorithm.
	'''

	if algorithm == 'kmeans':
		return kmeans_iteration_one(data, centroids, distance)
	elif algorithm == 'kmedians':
		return kmedians_iteration_one(data, centroids, distance)
	else:
		print 'Unavailable algorithm.\n'
		sys.exit(1)


def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument(dest='input', type=str, help='Dataset filename')
	parser.add_argument('-a', '--algorithm', dest='algorithm', type=str, help='Algorithm to use - {kmeans, kmedians}. Default: kmeans', default='kmeans')
	parser.add_argument('-i', '--init', '--initialization', dest='init', type=str, default='forgy', help='The initialization algorithm to be used - {forgy, kmeans++}. Default: forgy')
	parser.add_argument('-o', '--output', dest='output', type=str, help='Output filename. If not provided, centroids are not saved.')
	parser.add_argument('-m', '--iter', '--maxiter', dest='maxiter', type=int, default=1000, help='Maximum number of iterations of the algorithm to perform (may stop earlier if convergence is achieved). Default: 1000')
	parser.add_argument('-e', '--eps', '--epsilon', dest='epsilon', type=float, default=1e-3, help='Minimum distance the cluster centroids move b/w two consecutive iterations for the algorithm to continue. Default: 1e-3')
	parser.add_argument('-k', '--k', dest='k', type=int, default=8, help='The number of clusters to use. Default: 8')
	parser.add_argument('-s', '--seed', dest='seed', type=int, default=0, help='The RNG seed. Default: 0')
	parser.add_argument('-n', '--numexperiments', dest='numexperiments', type=int, default=1, help='The number of experiments to run. Default: 1')
	parser.add_argument('--outliers',dest='outliers',default=False,action='store_true',help='Flag for visualizing data without outliers. If provided, outliers are not plotted.')
	parser.add_argument('--verbose',dest='verbose',default=False,action='store_true',help='Turn on verbose.')
	_a = parser.parse_args()

	args = {}
	for a in vars(_a):
		args[a] = getattr(_a, a)

	if _a.algorithm.lower() in ['kmeans', 'means', 'k-means']:
		args['algorithm'] = 'kmeans'
		args['dist'] = distance_euclidean
	elif _a.algorithm.lower() in ['kmedians', 'medians', 'k-medians']:
		args['algorithm'] = 'kmedians'
		args['dist'] = distance_manhattan
	else:
		print 'Unavailable algorithm.\n'
		parser.print_help()
		sys.exit(1)

	if _a.init.lower() in ['k++', 'kplusplus', 'kmeans++', 'kmeans', 'kmeansplusplus']:
		args['init'] = initialization_kmeansplusplus
	elif _a.init.lower() in ['forgy', 'frogy']:
		args['init'] = initialization_forgy
	else:
		print 'Unavailable initialization function.\n'
		parser.print_help()
		sys.exit(1)

	print '-'*40 + '\n'
	print 'Arguments:'
	pprint(args)
	print '-'*40 + '\n'
	return args


def visualize_data(data, all_centroids, args):
	print 'Visualizing...'
	centroids = all_centroids[-1]
	k = args['k']
	distance = args['dist']
	clusters = [[] for _ in range(k)]
	for point in data:
		dlist = [distance(point, centroid) for centroid in centroids]
		clusters[argmin(dlist)].append(point)

	# plot each point of each cluster
	colors = cycle('rgbwkcmy')

	for c, points in zip(colors, clusters):
		x = [p[0] for p in points]
		y = [p[1] for p in points]
		plt.scatter(x,y, c=c)

	if not args['outliers']:
		# plot each cluster centroid
		colors = cycle('krrkgkgr')
		colors = cycle('rgbkkcmy')

		for c, clusterindex in zip(colors, range(k)):
			x = [iteration[clusterindex][0] for iteration in all_centroids]
			y = [iteration[clusterindex][1] for iteration in all_centroids]
			plt.plot(x,y, '-x', c=c, linewidth='1', mew=15, ms=2)
	plt.show()


def visualize_performance(data, all_centroids, distance):
	if distance == distance_euclidean:
		errors = [performance_SSE(data, centroids, distance) for centroids in all_centroids]
		ylabel = 'Sum Squared Error'
	else:
		errors = [performance_L1(data, centroids, distance) for centroids in all_centroids]
		ylabel = 'L1-norm Error'
	plt.plot(range(len(all_centroids)), errors)
	plt.title('Performance plot')
	plt.xlabel('Iteration')
	plt.ylabel(ylabel)
	plt.show()


if __name__ == '__main__':

	args = parse()
	# Read data
	data = readfile(args['input'])
	print 'Number of points in input data: {}\n'.format(len(data))
	verbose = args['verbose']

	totalerror = 0
	totaliter = 0

	for experiment in range(args['numexperiments']):
		print 'Experiment: {}'.format(experiment+1)
		random.seed(args['seed'] + experiment)
		print 'Seed: {}'.format(args['seed'] + experiment)

		# Initialize centroids
		centroids = []
		if args['init'] == initialization_forgy:
			centroids = args['init'](data, args['k'])  # Forgy doesn't need distance metric
		else:
			centroids = args['init'](data, args['dist'], args['k'])

		if verbose:
			print 'centroids initialized to:'
			print centroids
			print ''

		# Run clustering algorithm
		all_centroids = iteration_many(data, centroids, args['dist'], args['maxiter'], args['algorithm'], args['epsilon'])

		if args['dist'] == distance_euclidean:
			error = performance_SSE(data, all_centroids[-1], args['dist'])
			error_str = 'Sum Squared Error'
		else:
			error = performance_L1(data, all_centroids[-1], args['dist'])
			error_str = 'L1-norm Error'
		totalerror += error
		totaliter += len(all_centroids)-1
		print '{}: {}'.format(error_str, error)
		print 'Number of iterations till termination: {}'.format(len(all_centroids)-1)
		print 'Convergence achieved: {}'.format(hasconverged(all_centroids[-1], all_centroids[-2]))

		if verbose:
			print '\nFinal centroids:'
			print all_centroids[-1]
			print ''

	print '\n\nAverage error: {}'.format(float(totalerror)/args['numexperiments'])
	print 'Average number of iterations: {}'.format(float(totaliter)/args['numexperiments'])

	if args['numexperiments'] == 1:
		# save the result
		if 'output' in args and args['output'] is not None:
			writefile(args['output'], all_centroids[-1])

		# If the data is 2-d and small, visualize it.
		if len(data) < 5000 and len(data[0]) == 2:
			if args['outliers']:
				visualize_data(data[2:], all_centroids, args)
			else:
				visualize_data(data, all_centroids, args)

		visualize_performance(data, all_centroids, args['dist'])
