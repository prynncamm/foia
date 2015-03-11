# from copy import deepcopy
from glob import iglob

import math
import os
import yaml

import numpy as np

# Threshold set to 30ish percentile of scores
THRESHOLD = np.float64(0.940983)

"""
This script uses the Term Frequency–Inverse Document Frequency statistic
(tf-idf) to eliminate the keywords that score in the bottom 25th percentile
for relevance.
"""


def update_keyword_agency_dict(keywords_agency, keywords, agency_name):
    """
    Generate/update a dictionary, which maps keywords to agencies
    """
    for keyword in keywords:
        if keyword in keywords_agency:
            keywords_agency[keyword].append(agency_name)
        else:
            keywords_agency[keyword] = [agency_name]


def extract_agency_keywords(agency, agency_keywords, keywords_agency):
    """
    Update the agency_keywords and keywords_agency dictionaries with data
    from a specific agency
    """

    keywords = agency.get('keywords')
    if keywords:
        update_keyword_agency_dict(
            keywords_agency=keywords_agency,
            keywords=keywords,
            agency_name=agency.get('name'))
        agency_keywords[agency.get('name')] = keywords

    for office in agency['departments']:
        keywords = office.get('keywords')
        if keywords:
            update_keyword_agency_dict(
                keywords_agency=keywords_agency,
                keywords=keywords,
                agency_name=office.get('name'))
            agency_keywords[office.get('name')] = keywords


def get_keyword_dicts(glob_path):
    """
    Collect a dictionary that maps keywords to agencies and a dictionary that
    maps agencies to keywords
    """

    agency_keywords = {}
    keywords_agency = {}

    for filename in iglob(glob_path):
        with open(filename) as f:
            extract_agency_keywords(
                agency=yaml.load(f.read()),
                agency_keywords=agency_keywords,
                keywords_agency=keywords_agency)

    return agency_keywords, keywords_agency


def get_tf_idf(keyword, keywords, agency, agencies, total_agencies):
    """
    Compute tf_idf score of a given keyword currently tf (term frequency) has
    log weighting and idf is a simple inverse frequency.

    For more info http://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """

    idf = math.log(total_agencies / len(keywords.get(keyword)))
    tf = math.log(1 + agencies.get(agency).count(keyword))
    tf_idf = tf * idf
    return tf_idf


def clean_keywords(keywords, scores):
    """ Removes any keywords that have less than the median tfidf score """

    threshold = THRESHOLD
    if len(scores) > 0:
        if not threshold:
            threshold = np.median(scores)
        keywords = np.array(keywords)
        keywords = keywords[scores > threshold].tolist()
    keywords.sort()
    return keywords


def apply_tf_idf(agency, agencies, keywords, total_agencies):
    """
    Applies tfidf to score keywords and returns updated keywords
    """

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
                tf_idf_scores.append(tf_idf)
            office['keywords'] = clean_keywords(
                keyword_set, tf_idf_scores)
    return agency


def write_yaml(filename, data):
    """ Exports the updated yaml file """

    with open(filename, 'w') as f:
        f.write(yaml.dump(
            data, default_flow_style=False, allow_unicode=True))


def updated_yamls(glob_path):
    """ Updates the yaml files with cleaned keywords """

    agencies, keywords = get_keyword_dicts(glob_path)
    total_agencies = len(agencies.keys())

    for filename in iglob(glob_path):
        with open(filename) as f:
            agency = apply_tf_idf(
                agency=yaml.load(f.read()),
                agencies=agencies,
                keywords=keywords,
                total_agencies=total_agencies)
        write_yaml(filename=filename, data=agency)


if __name__ == "__main__":
    updated_yamls("data" + os.sep + "*.yaml")
