from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
import heapq
import math
from InvertedIndex import InvertedIndex
from MultiFileReader import MultiFileReader

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

bucket_name = 'wikidata_208373274'
client = storage.Client()
bucket = client.bucket(bucket_name)
#
# bins_prefixes = ['title_index/title_index_bins/', 'body_index/body_index_bins/', 'anchor_index/anchor_index_bins/']
# for prefix in bins_prefixes:
#     bins_blobs = bucket.list_blobs(prefix=prefix)
#     for blob in bins_blobs:
#         if '.bin' in blob.name:
#             dir_path = os.path.dirname( prefix)
#             os.makedirs(dir_path, exist_ok=True)
#             blob.download_to_filename(blob.name)
#
# pkl_prefixes = ['title_index/title_index.pkl', 'body_index/body_index.pkl', 'anchor_index/anchor_index.pkl']
# for prefix in pkl_prefixes:
#     pkl_blob = bucket.get_blob(prefix)
#     pkl_blob.download_to_filename(pkl_blob.name)
#
# page_view_blob = bucket.get_blob('doc_id_to_pageview_dict.pkl')
# page_view_blob.download_to_filename(page_view_blob.name)
#
# page_rank_blob = bucket.get_blob('doc_id_to_pagerank_dict.pkl')
# page_rank_blob.download_to_filename(page_rank_blob.name)
#
# doc_id_to_title_blob = bucket.get_blob('doc_id_to_title.pkl')
# doc_id_to_title_blob.download_to_filename(doc_id_to_title_blob.name)


with open('title_index/title_index.pkl', 'rb') as f:
    title_inverted_index = pickle.loads(f.read())
with open('body_index/body_index.pkl', 'rb') as f:
    body_inverted_index = pickle.loads(f.read())
with open('anchor_index/anchor_index.pkl', 'rb') as f:
    anchor_inverted_index = pickle.loads(f.read())
with open('doc_id_to_pagerank_dict.pkl', 'rb') as f:
    doc_id_to_pagerank_dict = pickle.loads(f.read())
with open('doc_id_to_pageview_dict.pkl', 'rb') as f:
    doc_id_to_pageview_dict = pickle.loads(f.read())
with open('doc_id_to_title.pkl', 'rb') as f:
    doc_id_to_title = pickle.loads(f.read())

stemmer = PorterStemmer()
def process_text(id, text, use_stemming=False):
    if use_stemming:
        list_of_tokens = list()
        for token in RE_WORD.finditer(text.lower()):
            stemmed_token = stemmer.stem(token.group())
            if stemmed_token not in all_stopwords:
                list_of_tokens.append(stemmed_token)
    else:
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return id, list_of_tokens


def search_long_query(query, k_to_return, cosine_title_result, cosine_body_result, cosine_anchor_result,
                      BOOLEAN_TITLE_WEIGHT, COSINE_TITLE_WEIGHT, COSINE_BODY_WEIGHT, COSINE_ANCHOR_WEIGHT,
                      PAGE_RANK_WEIGHT):
    BOOLEAN_TITLE_WEIGHT = 0

    COSINE_TITLE_WEIGHT = 0.1
    COSINE_BODY_WEIGHT = 0.35
    COSINE_ANCHOR_WEIGHT = 0.55

    set_of_all_doc_ids = set(cosine_title_result.keys()).union(set(cosine_body_result.keys()),
                                                               set(cosine_anchor_result.keys()))

    merged_scores_heapq = list()
    for doc_id in set_of_all_doc_ids:
        cosine_title_partial_score = COSINE_TITLE_WEIGHT * cosine_title_result.get(doc_id, 0)
        cosine_body_partial_score = COSINE_BODY_WEIGHT * cosine_body_result.get(doc_id, 0)
        cosine_anchor_partial_score = COSINE_ANCHOR_WEIGHT * cosine_anchor_result.get(doc_id, 0)

        merged_score = cosine_title_partial_score + cosine_body_partial_score + cosine_anchor_partial_score
        heapq.heappush(merged_scores_heapq, (-1 * merged_score, doc_id))
    sorted_top_num_of_docs = heapq.nsmallest(k_to_return, merged_scores_heapq)

    merged_scores_dict = {doc_id: ((-1 * negative_merged_score), doc_id_to_pagerank_dict.get(doc_id, 0)) for
                          negative_merged_score, doc_id in sorted_top_num_of_docs}
    sum_page_rank = max(sum([tup[1] for tup in merged_scores_dict.values()]), 1)
    top_scores_heapq_sum = list()
    for doc_id, merged_score_and_page_rank in merged_scores_dict.items():
        merged_score, page_rank = merged_score_and_page_rank
        heapq.heappush(top_scores_heapq_sum,
                       (-1 * (merged_score + (PAGE_RANK_WEIGHT * page_rank / sum_page_rank)), doc_id))

    top_11_results_sum = heapq.nsmallest(11, top_scores_heapq_sum)
    return [doc_id for negative_merged_score, doc_id in top_11_results_sum]


def search_common_query(query, k_to_return, boolean_title_result, cosine_title_result, cosine_body_result,
                        cosine_anchor_result, BOOLEAN_TITLE_WEIGHT, COSINE_TITLE_WEIGHT, COSINE_BODY_WEIGHT,
                        COSINE_ANCHOR_WEIGHT, PAGE_RANK_WEIGHT, docs_with_exact_title):
    SPECIAL_COSINE_ANCHOR_WEIGHT = COSINE_ANCHOR_WEIGHT
    SPECIAL_PAGE_RANK_WEIGHT = PAGE_RANK_WEIGHT

    weight_to_reduce_from_cosine_title_weight = 0.75 * COSINE_ANCHOR_WEIGHT
    SPECIAL_COSINE_ANCHOR_WEIGHT -= weight_to_reduce_from_cosine_title_weight
    SPECIAL_PAGE_RANK_WEIGHT += weight_to_reduce_from_cosine_title_weight

    set_of_all_doc_ids = set(boolean_title_result.keys()).union(set(cosine_title_result.keys()),
                                                                set(cosine_body_result.keys()),
                                                                set(cosine_anchor_result.keys()))

    merged_scores_heapq = list()
    for doc_id in set_of_all_doc_ids:
        boolean_title_partial_score = BOOLEAN_TITLE_WEIGHT * boolean_title_result.get(doc_id, 0)
        cosine_title_partial_score = COSINE_TITLE_WEIGHT * cosine_title_result.get(doc_id, 0)
        cosine_body_partial_score = COSINE_BODY_WEIGHT * cosine_body_result.get(doc_id, 0)
        cosine_anchor_partial_score = COSINE_ANCHOR_WEIGHT * cosine_anchor_result.get(doc_id, 0)
        if doc_id in docs_with_exact_title:
            cosine_anchor_partial_score = SPECIAL_COSINE_ANCHOR_WEIGHT * cosine_anchor_result.get(doc_id, 0)
        merged_score = boolean_title_partial_score + cosine_title_partial_score + cosine_body_partial_score + cosine_anchor_partial_score
        heapq.heappush(merged_scores_heapq, (-1 * merged_score, doc_id))

    sorted_top_num_of_docs = heapq.nsmallest(k_to_return, merged_scores_heapq)
    merged_scores_dict = {doc_id: (
    (-1 * negative_merged_score), doc_id_to_pagerank_dict.get(doc_id, 0) if doc_id in docs_with_exact_title else 0) for
                          negative_merged_score, doc_id in sorted_top_num_of_docs}

    sum_page_rank = max(sum([tup[1] for tup in merged_scores_dict.values()]), 1)
    top_scores_heapq_sum = list()
    for doc_id, merged_score_and_page_rank in merged_scores_dict.items():
        merged_score, page_rank = merged_score_and_page_rank
        if doc_id in docs_with_exact_title:
            heapq.heappush(top_scores_heapq_sum,
                           (-1 * (merged_score + (SPECIAL_PAGE_RANK_WEIGHT * page_rank / sum_page_rank)), doc_id))
        else:
            heapq.heappush(top_scores_heapq_sum,
                           (-1 * (merged_score + (PAGE_RANK_WEIGHT * page_rank / sum_page_rank)), doc_id))

    top_11_results_sum = heapq.nsmallest(11, top_scores_heapq_sum)
    return [doc_id for negative_merged_score, doc_id in top_11_results_sum]


def search_all(query):
    k_to_read = 15000
    k_to_return = 10000

    BOOLEAN_TITLE_WEIGHT = 0.25
    COSINE_TITLE_WEIGHT = 0.05
    COSINE_BODY_WEIGHT = 0.35
    COSINE_ANCHOR_WEIGHT = 0.35
    PAGE_RANK_WEIGHT = 0

    processed_query_tokens_list = process_text(0, query)[1]
    processed_query_tokens_list = [token for token in processed_query_tokens_list if token in title_inverted_index.df]
    num_of_tokenized_words = len(processed_query_tokens_list)
    if num_of_tokenized_words == 0:
        return []

    title_boolean_results = title_inverted_index.search_title_boolean(query, k_to_return, k_to_read)
    boolean_title_result = dict(
        [(doc_id, title_boolean_score / num_of_tokenized_words) for doc_id, title_boolean_score in
         title_boolean_results])

    docs_with_exact_title = [doc_id for doc_id, boolean_title_score in boolean_title_result.items() if
                             boolean_title_score == len(processed_query_tokens_list) / num_of_tokenized_words]
    num_of_docs_with_exact_title = len(docs_with_exact_title)

    cosine_title_result = dict(title_inverted_index.search_title_cosine(query, k_to_return, k_to_read))
    cosine_body_result = dict(body_inverted_index.search_cossim(query, k_to_return, k_to_read))
    cosine_anchor_result = dict(anchor_inverted_index.search_anchor_by_cosine(query, k_to_return, k_to_read))

    if num_of_tokenized_words > 1:
        if num_of_docs_with_exact_title == 0:
            return search_long_query(query, k_to_return, cosine_title_result, cosine_body_result, cosine_anchor_result,
                                     BOOLEAN_TITLE_WEIGHT, COSINE_TITLE_WEIGHT, COSINE_BODY_WEIGHT,
                                     COSINE_ANCHOR_WEIGHT, PAGE_RANK_WEIGHT)
        elif num_of_docs_with_exact_title > (len(boolean_title_result) * 0.01):
            return search_common_query(query, k_to_return, boolean_title_result, cosine_title_result,
                                       cosine_body_result, cosine_anchor_result, BOOLEAN_TITLE_WEIGHT,
                                       COSINE_TITLE_WEIGHT, COSINE_BODY_WEIGHT, COSINE_ANCHOR_WEIGHT, PAGE_RANK_WEIGHT,
                                       docs_with_exact_title)

    set_of_all_doc_ids = set(cosine_title_result.keys()).union(set(cosine_body_result.keys()),
                                                               set(cosine_anchor_result.keys()))
    merged_scores_heapq = list()
    for doc_id in set_of_all_doc_ids:
        boolean_title_partial_score = BOOLEAN_TITLE_WEIGHT * boolean_title_result.get(doc_id, 0)
        cosine_title_partial_score = COSINE_TITLE_WEIGHT * cosine_title_result.get(doc_id, 0)
        cosine_body_partial_score = COSINE_BODY_WEIGHT * cosine_body_result.get(doc_id, 0)
        cosine_anchor_partial_score = COSINE_ANCHOR_WEIGHT * cosine_anchor_result.get(doc_id, 0)

        merged_score = boolean_title_partial_score + cosine_title_partial_score + cosine_body_partial_score + cosine_anchor_partial_score
        heapq.heappush(merged_scores_heapq, (-1 * merged_score, doc_id))
    sorted_top_num_of_docs = heapq.nsmallest(k_to_return, merged_scores_heapq)

    merged_scores_dict = {doc_id: ((-1 * negative_merged_score), doc_id_to_pagerank_dict.get(doc_id, 0)) for
                          negative_merged_score, doc_id in sorted_top_num_of_docs}
    sum_page_rank = max(sum([tup[1] for tup in merged_scores_dict.values()]), 1)
    top_scores_heapq_sum = list()
    for doc_id, merged_score_and_page_rank in merged_scores_dict.items():
        merged_score, page_rank = merged_score_and_page_rank
        heapq.heappush(top_scores_heapq_sum,
                       (-1 * (merged_score + (PAGE_RANK_WEIGHT * page_rank / sum_page_rank)), doc_id))

    top_11_results_sum = heapq.nsmallest(11, top_scores_heapq_sum)
    return [doc_id for negative_merged_score, doc_id in top_11_results_sum]

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    doc_id_to_scores = search_all(query)
    res = [(doc_id, doc_id_to_title.get(doc_id, 'unknown')) for doc_id in doc_id_to_scores]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    doc_id_to_scores = body_inverted_index.search_cossim(query, 100, 0)
    res = [(doc_id, doc_id_to_title.get(doc_id, 'unknown')) for doc_id, score in doc_id_to_scores]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    doc_id_to_scores = title_inverted_index.search_title_boolean(query, 0, 0)
    res = [(doc_id, doc_id_to_title.get(doc_id, 'unknown')) for doc_id, score in doc_id_to_scores]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    doc_id_to_scores = anchor_inverted_index.search_anchor_nir(query)
    res = [(doc_id, doc_id_to_title.get(doc_id, 'unknown')) for doc_id, score in doc_id_to_scores]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [doc_id_to_pagerank_dict.get(doc_id, 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [doc_id_to_pageview_dict.get(doc_id, 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


