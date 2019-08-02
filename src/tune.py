# implementation based on 
# 1. https://github.com/nmslib/nmslib/blob/master/similarity_search/lshkit/tools/fitdata.cpp
# 2. https://github.com/nmslib/nmslib/blob/master/similarity_search/lshkit/tools/mplsh-tune.cpp

import numpy as np
import random
import math
from queue import PriorityQueue


def fit():
	f = open(DATASET_FILE)
	elem_size = np.fromfile(f, dtype=np.uint32, count=1)
	size = np.fromfile(f, dtype=np.uint32, count=1).item()
	dims = np.fromfile(f, dtype=np.uint32, count=1)
	print(f'elem_size: {elem_size}, size: {size}, dims: {dims}')
	vectors = []
	for idx in range(size):
		vector = np.fromfile(f, dtype=np.single, count=dims.item())
		vectors.append(vector)
	f.close()

	sample_ids = list(range(0, len(vectors)))
	random.shuffle(sample_ids)	
	sample_ids = sample_ids[:FIT_N]

	# sampe FIT_P pairs of points
	gm = 0
	gg = 0	
	for idx in range(FIT_P):
		i = np.random.randint(0, FIT_N)
		j = np.random.randint(0, FIT_N)
		if i == j:
			continue
		dist = np.linalg.norm(vectors[sample_ids[i]] - vectors[sample_ids[j]]) #l2 distance
		logdist = math.log(dist)
		gm += dist
		gg += logdist
	gm /= FIT_P;
	gg /= FIT_P;
	gg = math.exp(gg);

	# sample queries
	Q = FIT_Q if FIT_Q <= FIT_N else FIT_N
	if FIT_K > (FIT_N - Q):
		K = FIT_N - Q
	else:
		K = FIT_K
	# generate random non-duplicate Q queries from sample_ids
	query_ids = []
	while len(query_ids) < Q:
		query_id = np.random.randint(0, FIT_N)
		if query_id not in query_ids:
			query_ids.append(query_id)
	# execute queries
	yM = []
	yG = []
	topks = []
	X1 = []
	X2 = []
	X3 = []
	for q in range(Q):
		topks.append([])
	l = 0
	while l < FIT_F:
		i = l
		while i < FIT_N:
			q = 0
			while q < Q:
				if i != query_ids[q]:
					dist = np.linalg.norm(vectors[sample_ids[query_ids[q]]] - vectors[sample_ids[i]])
					append_top_k(topks[q], (dist,i), K)
				q+=1
			i+=FIT_F
		M = []
		G = []
		for k in range(K):
			M.append(0)
			G.append(0)
		for q in range(Q):
			for k in range(K):
				M[k] += topks[q][k][0]
				G[k] += math.log(topks[q][k][0])
		for k in range(K):
			M[k] = math.log(M[k]/Q);
			G[k] = G[k]/Q;
			yM.append(M[k])
			yG.append(G[k])
			X1.append(1.0)
			X2.append(math.log(size * (l+1)/ FIT_F))
			X3.append(math.log(k+1))
		l+=1

	# solve multiple linear regression equation
	X = np.array((X1,X2,X3)).T
	pM = np.linalg.lstsq(X, yM)[0]
	pG = np.linalg.lstsq(X, yG)[0]
	print(f'gm: {gm}, gg: {gg}')
	print(pM)
	print(pG)

	garr = np.array([gm, gg], dtype=np.float32)
	pmarr = np.array([pM[0], pM[1], pM[2]], dtype=np.float32)
	pgarr = np.array([pG[0], pG[1], pG[2]], dtype=np.float32)
	f = open(PARAMS_FILE, "w")
	garr.tofile(f)
	pmarr.tofile(f)
	pgarr.tofile(f)
	f.close()


def append_top_k(l, el, max_size):
	(dist, i) = el
	if len(l) == max_size:
		(last_dist, last_i) = l[max_size - 1]
		if dist > last_dist:
			return
		else:
			l.pop()
	l.append((dist, i))
	l.sort()


def tune():
	# build expected model
	f = open(PARAMS_FILE)
	(M, G) = np.fromfile(f, dtype=np.float32, count=2)
	(a_M, b_M, c_M) = np.fromfile(f, dtype=np.float32, count=3)
	(a_G, b_G, c_G) = np.fromfile(f, dtype=np.float32, count=3)
	f.close()
	print(M, G)
	print(a_M, b_M, c_M)
	print(a_G, b_G, c_G)
	temp_m = math.exp(a_M) * math.pow(TUNE_N, b_M)
	temp_g = math.exp(a_G) * math.pow(TUNE_N, b_G)
	topk_distrs = []
	for k in range(TUNE_K):
		m = temp_m * math.pow(k + 1, c_M);
        g = temp_g * math.pow(k + 1, c_G);

        # check https://waterprogramming.wordpress.com/2016/08/18/root-finding-in-matlab-r-python-and-c/
        gamma_dist = np.random.gamma(shape = k, scale = m/k)
		topk_distrs.append(gamma_dist)


	MIN_L = 1
	MAX_L = 20
	MIN_T = 1
	MAX_T = 200
	MIN_M = 1 
	MAX_M = 64 # max number of hash tables
	MIN_W = 0.01
	MAX_W = 10	# max width length
	NUM_W = 400
	DELTA_W = (MAX_W - MIN_W) / NUM_W

	intervals = []
	intervals.append((MIN_L, MAX_L + 1))
	intervals.append((MIN_T, MAX_T + 1))
	intervals.append((0, MAX_M - MIN_M + 1)) # number of hash table
	intervals.append((0, NUM_W + 1)) # width length

	

	intervals[0] = (TUNE_L, TUNE_L + 1)
	intervals[1] = (TUNE_T, TUNE_T + 1)

	int m = intervals[2][0]
	int end_m = intervals[2][1]
	'''
	while m < end_m:
		intervals[2][0] = m
        intervals[2][1] = m + 1
        # finding optimal W through binary search
	'''

def recall_K(l, t, m, w):




# MAIN 
DATASET_FILE = 'data/audio/audio.data'
QUERY_FILE = 'data/audio/audio.query'
PARAMS_FILE = 'data/audio/audio.params'

FIT_N = 2000 # number of points to use
FIT_P = 50000 # number of pairs to sample
FIT_Q = 1000 # number of queries to sample 
FIT_K = 100 # search for K nearest neighbors
FIT_F = 10 # divide the sample to F folds

TUNE_N = 54387 # dataset size
TUNE_RECALL = 0.9 # desired recall
TUNE_L = 8 # number of hash tables
TUNE_T = 20 # number of bins to probe during query
TUNE_K = FIT_K # number of top K docs to find

#fit()
tune()