import sys
import pandas as pd
import numpy as np
import os
import pickle
import networkx as nx

path = os.getcwd()


kg = pd.read_csv(f'{path}/ml1m/preprocessed/kg_final.txt', sep="\t")

kg_grouped_head = kg.groupby('entity_head').agg({'entity_tail': lambda x: list(x), 
                                                                'relation': lambda x: list(x)})
kg_grouped_head_old = kg.groupby('entity_head')["entity_tail"].agg(list)


kg_grouped_tail = kg.groupby('entity_tail').agg({'entity_head': lambda x: list(x), 
                                                                'relation': lambda x: list(x)})
kg_grouped_tail_old = kg.groupby('entity_tail')["entity_head"].agg(list)

graph = kg_grouped_head_old.to_frame()
inverse_graph = kg_grouped_tail_old.to_frame()

graph_new = kg_grouped_head.apply(lambda row: list(zip(row['entity_tail'], row['relation'])), axis=1)
inverse_graph_new = kg_grouped_tail.apply(lambda row: list(zip(row['entity_head'], row['relation'])), axis=1)

ratings = pd.read_csv(f'{path}/ml1m/preprocessed/ratings.txt', sep='\t')
movies_map = pd.read_csv(f'{path}/ml1m/preprocessed/products.txt', sep='\t', index_col='new_id')
i2kg = pd.read_csv(f'{path}/ml1m/preprocessed/i2kg_map.txt', sep='\t')

old_movies = pd.read_csv(f'{path}/ml1m/movies.dat',  sep="::", names=["movie_id", "movie_name", "genre"], header=None, encoding='latin-1')
old_movies_edited = old_movies
old_movies_edited['genre'] = old_movies_edited['genre'].str.split('|').to_list()


e = pd.read_csv(f'{path}/ml1m/preprocessed/e_map.txt', sep="\t", index_col='entity_id')
r = pd.read_csv(f'{path}/ml1m/preprocessed/r_map.txt', sep="\t", index_col='relation_id')

def find_shortest_path(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])

    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r != 12 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r != 12:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_director(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])


    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break


        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 18 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 18:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_actor(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 10 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 10:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_writter(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 16 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 16:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_production_company(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 1 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 1:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_category(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 3 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 3:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev

def find_cinematographer(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 0 :#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 0:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev


def find_related(src, dest):

    vis_to_indx_graph = dict(enumerate(graph_new.index.values.flatten(), 0))
    vis_to_indx_graph = dict([(value, key) for key, value in vis_to_indx_graph.items()])

    vis_to_indx_inverse = dict(enumerate(inverse_graph_new.index.values.flatten(), 0))
    vis_to_indx_inverse = dict([(value, key) for key, value in vis_to_indx_inverse.items()])



    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    stack1.append(src)
    
    while len(stack1) > 0:


        d = stack1.pop(0)

        if d == dest:
            break
        

        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]] and r == 12:#and r != 12 and r != 3 and r != 11 and r != 1
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r, e)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]] and r == 12:#and r != 12 and r != 3 and r != 11 and r != 1
                  stack1.append(e)
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r, e)


    return prev

def get_path(arr, from_mov, to_movie):

    if (len(arr) == 0):
        return -1;

    end = (to_movie, 5)
    

    relations = []

    while end[0] != from_mov:
        end = arr[end[0]]
        relations.append(end)

    if len(relations) <= 3:
        return relations;
    else:
        return -1

def get_rules(from_movie, to_movie):

    rules = {
        "cinematographer": [],
        "director": [],
        "actor": [],
        "writter": [],
        "production_company": [],
        "category": [],
        "realted": []
    }

    rules_dic = {}

    for key in (from_movie):
        rules_dic[key] = {}
        for j in (rules.keys()):
            rules_dic[key][j] = -1



    for i in from_movie:

        director = find_director(i, to_movie)
        actor = find_actor(i, to_movie)
        writter = find_writter(i, to_movie)
        production_company = find_production_company(i, to_movie)
        category = find_category(i, to_movie)
        cinematographer = find_cinematographer(i, to_movie)


        if(len(director) != 0 and to_movie in director):
            director = get_path(director, i)
            if director != -1:
                rules_dic[i]['director'] = director

        
        if(len(actor) != 0 and to_movie in actor):
            actor = get_path(actor, i)
            if actor != -1:
                rules_dic[i]["actor"] = actor



        if(len(writter) != 0 and to_movie in writter):
            writter = get_path(writter, i)
            if writter != -1:
                rules_dic[i].writter = writter

        if(len(production_company) != 0 and to_movie in production_company):
            production_company = get_path(production_company, i)
            if production_company != -1:
                rules_dic[i]['production_company'] =production_company

        
        if(len(category) != 0 and to_movie in category):
            category = get_path(category, i)
            if category != -1:
                rules_dic[i]['category'] = category

        if(len(cinematographer) != 0 and to_movie in cinematographer):
            cinematographer = get_path(cinematographer, i) 
            if cinematographer != -1:
                rules_dic[i]['cinematographer'] = cinematographer
        
    return rules, rules_dic


def print_rules(from_movie, rules_dic):
    for i in from_movie:
        print(i, rules_dic[i])



def find_all_paths(src, dest):
    visited1 = [False] * graph_new.shape[0]
    visited2 = [False] * inverse_graph_new.shape[0]

    stack1 = []
    prev = {}
    path = []
    paths = []
    stack1.append(src)
    
    while len(stack1) > 0:


        if len(stack1) > 3:
            stack1.pop()
            path.pop()
            continue


        d = stack1.pop()

        if d == dest:
            print(path)
            paths.append(path.copy())
            path.pop()
            continue

            
        if d in vis_to_indx_inverse:
          for e, r in inverse_graph_new.loc[d]:
           if not  visited1[vis_to_indx_graph[e]]:
                    path.append((d, r, e))
                    stack1.append(e)
                    visited1[vis_to_indx_graph[e]] = True;
                    prev[e] = (d, r)
        else:
           for e, r in graph_new.loc[d]:
            if not visited2[vis_to_indx_inverse[e]]:
                  stack1.append(e)
                  path.append((d, r, e))
                  visited2[vis_to_indx_inverse[e]] = True;
                  prev[e] = (d, r)

    return paths


relation_to_entity = {
        "http://dbpedia.org/ontology/cinematography": 'cinematographer',
        "http://dbpedia.org/property/productionCompanies": 'production_company',
        "http://dbpedia.org/property/composer": 'composer',
        "http://purl.org/dc/terms/subject": 'category',
        "http://dbpedia.org/ontology/starring": 'actor',
        "http://dbpedia.org/ontology/country": 'country',
        "http://dbpedia.org/ontology/wikiPageWikiLink": 'wikipage',
        "http://dbpedia.org/ontology/editing": 'editor',
        "http://dbpedia.org/property/producers": 'producer',
        "http://dbpedia.org/property/allWriting": 'writter',
        "http://dbpedia.org/ontology/director": 'director',
    }

relation_id_to_plain_name =  {
        "0" : "cinematography by the same cinematographer ",
        "1" : "produced by the same company ",
        "2" : "composed_by",
        "3" : "belong to the same category ",
        "10": "starred by the same leading actor ",
        "11": "produced_in",
        "12": "movie_name related to recommended movie",
        "14": "edited_by",
        "15": "produced_by_producer",
        "16": "wrote by the same writer ",
        "18": "directed by the same director ",
    }

#ML1M ENTITIES
MOVIE = 'movie'
ACTOR = 'actor'
DIRECTOR = 'director'
PRODUCTION_COMPANY = 'production_company'
EDITOR = 'editor'
WRITTER = 'writter'
CINEMATOGRAPHER = 'cinematographer'
COMPOSER = 'composer'
COUNTRY = 'country'
AWARD = 'award'


def translate(relations):
    new_r = []
    e1 = relations[1][0]
    r1 = relations[0][1]
    e2 = relations[0][0]
    new_r.append((e.loc[e1]["entity_url"][28:], relation_id_to_plain_name[str(r1)], e.loc[e2]["entity_url"][28:]))

    return new_r

def translate_all(from_movie, rules, rules_dic):
    for key in (from_movie):
        for j in (rules.keys()):
            if rules_dic[key][j] != -1:
                rules_dic[key][j] = translate(rules_dic[key][j])

def get_most_similar_user(user_id):
    with open('graph.pickle', 'rb') as file:
            G = pickle.load(file)
    user_index = ratings['uid'].unique().tolist().index(user_id)
    pagerank = nx.pagerank(G, weight='weight')
    most_similar_user_index = max(pagerank.items(), key=lambda x: x[1])[0]
    most_similar_user_id = ratings['uid'].unique()[most_similar_user_index]
    return most_similar_user_id

def get_movie_explanation(uid, pid):

    user_id = uid
    user_ratings = ratings[(ratings['uid'] == user_id) & (ratings['rating'] >= 3)]
    user_movies = user_ratings["pid"]
    pid = pid

    to_movie = 0


    raw_dataset_id = movies_map.iloc[pid]["raw_dataset_id"]
    to_movie = i2kg[i2kg['dataset_id'] == raw_dataset_id]['entity_id'].iloc[0]

    genres = {
        'Animation': 0,
        'Adventure': 0,
        'Children\'s': 0, 
        'Fantasy': 0, 
        'Comedy': 0, 
        'Romance': 0, 
        'Drama': 0, 
        'Thriller': 0, 
        'Action': 0, 
        'Crime': 0, 
        'Musical': 0, 
        'War': 0, 
        'History': 0, 
        'Documentary': 0, 
        'Mystery': 0,
        'Sci-Fi': 0,
        'Film-Noir': 0,
        'Horror': 0
    }

    from_movie = []

    for i in user_movies:
        raw_dataset_id = movies_map.iloc[i]["raw_dataset_id"]
        entity_id = i2kg[i2kg['dataset_id'] == raw_dataset_id]['entity_id'].iloc[0]
        from_movie.append(entity_id)
        for j in old_movies_edited[old_movies_edited['movie_id'] == raw_dataset_id]['genre'].iloc[0]:
            genres[j] += 1



    for i in from_movie:
        if kg[kg['entity_head'] == i]['entity_head'].count() == 0:
            from_movie.remove(i)

    #explaination number 1 favorite genres 
    top_genres = sorted(genres, key=genres.get, reverse=True)[:3]

    #explaination number 2 user similarity
    most_similar_user = get_most_similar_user(uid)

    #explaination number 3 movie knowledge graph rules
    rules, rules_dic = get_rules(from_movie, to_movie)
    translated_rules = translate_all(from_movie, rules, rules_dic)

    print_rules(from_movie, translated_rules)
