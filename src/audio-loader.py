import numpy as np
import requests
import json
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def load_train_data_to_es():
	client = Elasticsearch(http_auth=('elastic', 'password'))
	print("Creating the " + INDEX_NAME + " index.")
	client.indices.delete(index=INDEX_NAME, ignore=[404])
	with open(INDEX_MAPPING_FILE) as file:
		source = file.read().strip()
		client.indices.create(index=INDEX_NAME, body=source, request_timeout=120)

	with open(DATASET_FILE) as f:
		elem_size = np.fromfile(f, dtype=np.uint32, count=1)
		size = np.fromfile(f, dtype=np.uint32, count=1)
		dims = np.fromfile(f, dtype=np.uint32, count=1)
		print(f'elem_size: {elem_size}, size: {size}, dims: {dims}')

		print("Indexing documents...")
		bulk_requests = []
		for idx in range(size.item()):
			vector = np.fromfile(f, dtype=np.single, count=dims.item())
			request = {
				'_op_type': 'index',
				'_index': INDEX_NAME, 
				'_id': idx,
	    		'vector': vector.tolist(),
	    		'my_dense_vector' : vector.tolist()
			}
			bulk_requests.append(request)
			if idx % BATCH_SIZE == 0:
			    bulk(client, bulk_requests)
			    bulk_requests = []	
		if bulk_requests:
			bulk(client, bulk_requests)
		client.indices.refresh(index=INDEX_NAME)
		print('Done indexing.')	

def query_test_data():
	# fill query_vectors
	query_vectors = []
	df = open(DATASET_FILE, "r")
	elem_size = np.fromfile(df, dtype=np.uint32, count=1)
	size = np.fromfile(df, dtype=np.uint32, count=1)
	dims = np.fromfile(df, dtype=np.uint32, count=1)
	for idx in range(NUMBER_OF_QUERIES):
		query_vector = np.fromfile(df, dtype=np.single, count=dims.item())
		query_vectors.append(query_vector.tolist())
	df.close()


	start_time = time.time()
	#<query ID> <K> <1st NN's ID> <distance> <2nd NN's ID> <distance> ... <Kth NN's ID> <distance>
	qf = open(QUERY_FILE, "r")
	average_common = 0
	max_recall_queries_num = 0 # number of queries that got all TOP_SIZE docs right
	for idx in range(NUMBER_OF_QUERIES):
		query_neighbors = []
		distances = []
		for idx2, el in enumerate(qf.readline().split()):
			if idx2 > 1:
				if (idx2 % 2 == 0):
					query_neighbors.append(int(el))
				else:
					distances.append(float(el))
		#is_display = True if idx == 0 else False
		is_display = False
		common = query(idx, query_vectors[idx], query_neighbors, distances, is_display)
		average_common += common
		if common == (TOP_SIZE - 1):
			max_recall_queries_num += 1
	qf.close()
	average_common = average_common / NUMBER_OF_QUERIES
	elapsed_time = (time.time() - start_time) * 1000
	print(f'recall: {average_common / (TOP_SIZE - 1)}, number of queries with 100% recall: {max_recall_queries_num}, time per query: {elapsed_time / NUMBER_OF_QUERIES} ms')		

def query(idx, query_vector, neighbors, distances, print_response = False):
	headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
	uri = 'http://localhost:9200/' + INDEX_NAME + "/_search"
	if QUERY_TYPE == 0: # knn
		ann_query = {
			"query" : {
				"script_score" : {
					"query" : {
       					"match_all": {}
       				},
       				"script" : {
       					"source": "1 / (1 + l2norm(params.queryVector, doc['my_dense_vector']))", 
       					"params" : {
       						"queryVector" : query_vector
       					}
       				}
				}
			}
		}
	elif QUERY_TYPE == 1: # ann
		ann_query = {
			"query" : {
				"ann": {
					"field": "vector",
					"query_vector": query_vector,
					"number_of_probes" : NUMBER_OF_PROBES
				} 
			}
		}
	else:
		ann_query = {  # ann with rescore
			"query" : {
				"ann": {
					"field": "vector",
					"query_vector": query_vector,
					"number_of_probes" : NUMBER_OF_PROBES
				} 
			},
			"rescore" : {
				"window_size" : QUERY_SIZE,
				"query" : {
					"query_weight" : 0,
         			"rescore_query_weight" : 1,
					"rescore_query" : {
						"script_score" : {
							"query" : {
	           					"match_all": {}
	           				},
	           				"script" : {
	           					"source": "1 / (1 + l2norm(params.queryVector, doc['my_dense_vector']))", 
	           					"params" : {
	           						"queryVector" : query_vector
	           					}
	           				}
						}
					}
				}
			}
		}
	request = json.dumps(ann_query)
	response = requests.get(
		uri, 
		data=request, 
		headers=headers, 
		auth=('elastic', 'password'),
		params={'size': QUERY_SIZE, '_source' : False},
	)
	common = 0
	if (response.status_code == requests.codes.ok):
		results = json.loads(response.text)
		hits_num = results['hits']['total']['value']
		max_score = results['hits']['max_score']
		doc_ids = [int(doc["_id"]) for doc in results['hits']['hits']]
		top_docs_ids = doc_ids[:TOP_SIZE]
		top_neighbors = neighbors[:TOP_SIZE]
		for idx2, doc_id in enumerate(top_neighbors):
			if int(doc_id) in top_docs_ids:
				common = common + 1
			else:
				if print_response:
					print(f'---not found! doc_id:{doc_id} distance: {distances[idx2]}')
		print(f'{idx} hits: {hits_num}, max_score: {max_score}, common: {common}')
		if print_response:
			print(query_vector)
			print(neighbors)
			print(distances)
			#print(response.text)
			data = [doc for doc in results['hits']['hits']]
			for doc in data:
				print(f'doc: {doc["_id"]} is {doc["_score"]}')
	return common	

# MAIN 
DATASET_FILE = 'data/audio/audio.data'
QUERY_FILE = 'data/audio/audio.query'
INDEX_MAPPING_FILE = 'data/audio/index.json'
INDEX_NAME = 'audio2'
BATCH_SIZE = 100

NUMBER_OF_QUERIES = 10000 #10000
NUMBER_OF_PROBES = 20
QUERY_SIZE = 500 + 1 # +1 to exclude the query doc itself
QUERY_TYPE = 2
if QUERY_TYPE == 2: #ann with rescore
	TOP_SIZE = 10 + 1
else:
	TOP_SIZE = QUERY_SIZE
#load_train_data_to_es()
query_test_data()