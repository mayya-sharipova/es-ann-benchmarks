import h5py
import numpy
import requests
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def load_train_data_to_es():
	client = Elasticsearch(http_auth=('elastic', 'password'))
	print("Creating the " + INDEX_NAME + " index.")
	client.indices.delete(index=INDEX_NAME, ignore=[404])
	with open(INDEX_MAPPING_FILE) as file:
		source = file.read().strip()
		client.indices.create(index=INDEX_NAME, body=source, request_timeout=120)

	dataset = h5py.File(DATASET_FILE, 'r')
	dimension = len(dataset['train'][0])  
	point_type = dataset.attrs.get('point_type', 'float')
	train_data = numpy.array(dataset['train'])
	vectors = numpy.ascontiguousarray(train_data, dtype=numpy.float32)
	print('got a train set of size (%d * %d)' % train_data.shape)
	dataset.close()
 	
	print("Indexing documents...")
	bulk_requests = []
	titles = []

	for idx, vector in enumerate(vectors):
		request = {
			'_op_type': 'index',
			'_index': INDEX_NAME, 
			'_id': idx,
    		'vector': vector.tolist()
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
	dataset = h5py.File(DATASET_FILE, 'r')
	test_data = numpy.array(dataset['test'])
	print('got a test set of size (%d * %d)' % test_data.shape)
	query_vectors = numpy.ascontiguousarray(test_data, dtype=numpy.float32)

	test_data = numpy.array(dataset['neighbors'])
	query_neighbors = numpy.ascontiguousarray(test_data, dtype=numpy.float32)
	dataset.close()

	headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
	uri = 'http://localhost:9200/' + INDEX_NAME + "/_aknnsearch"
	number_of_queries = 100
	average_common = 0
	for idx, query_vector in enumerate(query_vectors):
		if idx == number_of_queries:
			break
		if idx == 0:	
			average_common += query(query_vector.tolist(), query_neighbors[idx].tolist(), False)
		else:
			average_common += query(query_vector.tolist(), query_neighbors[idx].tolist())	
	average_common = average_common / number_of_queries
	print(f'recall: {average_common / 100}')		

def query(query_vector, neighbors, print_response = False):
	headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
	uri = 'http://localhost:9200/' + INDEX_NAME + "/_search"
	ann_query = {
		"query" : {
			"ann": {
				"field": "vector",
				"query_vector": query_vector,
				"number_of_probes" : 10
			} 
		}
	}
	request = json.dumps(ann_query)
	response = requests.get(
		uri, 
		data=request, 
		headers=headers, 
		auth=('elastic', 'password'),
		params={'size': 100, '_source' : False},
	)
	common = 0
	if (response.status_code == requests.codes.ok):
		results = json.loads(response.text)
		hits_num = results['hits']['total']['value']
		max_score = results['hits']['max_score']
		doc_ids = [int(doc["_id"]) for doc in results['hits']['hits']]
		for doc_id in neighbors:
			if int(doc_id) in doc_ids:
				common = common + 1


		print(f'hits: {hits_num}, max_score: {max_score}, common: {common}')
		if print_response:
			print(neighbors)
			print(response.text)
			data = [doc for doc in results['hits']['hits']]
			for doc in data:
				print(f'doc: {doc["_id"]} is {doc["_score"]}')
	return common			


# MAIN 
BATCH_SIZE = 50
DATASET_FILE = 'data/fashion-mnist/fashion-mnist-784-euclidean.hdf5'
INDEX_MAPPING_FILE = 'data/fashion-mnist/index.json'
INDEX_NAME = 'fashion-mnist'
load_train_data_to_es()
#query_test_data()

