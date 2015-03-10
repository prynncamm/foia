# from copy import deepcopy
from glob import iglob

import math
import os
import yaml

import numpy as np
from stop_words import get_stop_words

stop_words = get_stop_words('english')


def flatten_keywords(keywords):
    """
    Converts phases into keywords while removing stop words and parentheses
    """

    flattened_keywords = []
    if keywords:
        for keyword in keywords:
            split_words = list(set(keyword.lower().split(' ')))
            for word in split_words:
                word = word.strip("()")
                if word not in stop_words:
                    flattened_keywords.append(word)
    return flattened_keywords


def update_keyword_agency_dict(keywords_agency, keywords, agency_name):
    """
    Generates/updates a dictionary, which maps keywords to agencies
    """
    for keyword in keywords:
        if keyword in keywords_agency:
            keywords_agency[keyword].append(agency_name)
        else:
            keywords_agency[keyword] = [agency_name]


def get_keyword_dicts(glob_path):

    agency_keywords = {}
    keywords_agency = {}

    for filename in iglob(glob_path):
        with open(filename) as f:
            agency = yaml.load(f.read())

        keywords = flatten_keywords(agency.get('keywords'))
        if keywords:
            update_keyword_agency_dict(
                keywords_agency=keywords_agency,
                keywords=keywords,
                agency_name=agency.get('name'))
            agency_keywords[agency.get('name')] = keywords

        for office in agency['departments']:
            keywords = flatten_keywords(office.get('keywords'))
            if keywords:
                update_keyword_agency_dict(
                    keywords_agency=keywords_agency,
                    keywords=keywords,
                    agency_name=office.get('name'))
                agency_keywords[office.get('name')] = keywords
    return agency_keywords, keywords_agency


def get_tf_idf(keyword, keywords, agency, agencies, total_agencies):
    """
    Computes tf_idf score of a given keyword currently tf (term freqency) has
    no weighting and idf is a simple inverse frequency.

    for more info http://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """

    idf = math.log(total_agencies / len(keywords.get(keyword)))
    tf = agencies.get(agency).count(keyword)
    tf_idf = tf * idf
    return tf_idf


def clean_keywords(keywords, scores):

    if len(scores) > 1:
        median = np.median(scores)
        keywords = np.array(keywords)
        keywords = keywords[scores > median].tolist()

    return keywords


def apply_tf_idf(glob_path):

    agencies, keywords = get_keyword_dicts(glob_path)
    total_agencies = len(agencies.keys())

    for filename in iglob(glob_path):
        with open(filename) as f:
            agency = yaml.load(f.read())

        if 'keywords' in agency:
            tf_idf_scores = []
            keyword_set = list(set(agencies.get(agency['name'])))
            for keyword in keyword_set:
                tf_idf = get_tf_idf(
                    keyword=keyword,
                    keywords=keywords,
                    agency=agency['name'],
                    agencies=agencies,
                    total_agencies=total_agencies)
                tf_idf_scores.append(tf_idf)
            agency['keywords'] = clean_keywords(keyword_set, tf_idf_scores)

        for office in agency['departments']:
            if 'keywords' in office:
                keyword_set = list(set(agencies.get(office['name'])))
                tf_idf_scores = []
                for keyword in keyword_set:
                    tf_idf = get_tf_idf(
                        keyword=keyword,
                        keywords=keywords,
                        agency=office['name'],
                        agencies=agencies,
                        total_agencies=total_agencies)
                office['keywords'] = clean_keywords(keyword_set, tf_idf_scores)

        with open(filename, 'w') as new_file:
            new_file.write(yaml.dump(
                agency, default_flow_style=False, allow_unicode=True))


if __name__ == "__main__":

    apply_tf_idf("data" + os.sep + "*.yaml")
