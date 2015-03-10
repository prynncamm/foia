from unittest import TestCase

import clean_keywords


class CleanKeywordsTests(TestCase):

    def test_flatten_keywords(self):
        """ Verify that list items are flattend into a list of words """

        test_list = ['freedom of information', 'Information', '(United)']
        result = clean_keywords.flatten_keywords(test_list)

        self.assertEqual(len(result), 4)
        # Phrases broken into words
        self.assertEqual(result.count('information'), 2)
        # Stop words removed
        self.assertEqual(result.count('of'), 0)
        # Parentheses are removed and words in lowercase
        self.assertEqual(result.count('united'), 1)

    def test_update_keyword_agency_dict(self):
        """ Verify that a dictionary of keywords to agencies is updated """

        # Frist run
        keywords_agency = {}
        keywords = ['freedom', 'information']
        agency_name = 'Agency 1'
        clean_keywords.update_keyword_agency_dict(
            keywords_agency=keywords_agency,
            keywords=keywords,
            agency_name=agency_name)
        expected_output = {
            'information': ['Agency 1'],
            'freedom': ['Agency 1']
        }
        self.assertEqual(keywords_agency, expected_output)

        # Second run
        keywords = ['freedom', 'united']
        agency_name = 'Agency 2'
        clean_keywords.update_keyword_agency_dict(
            keywords_agency=keywords_agency,
            keywords=keywords,
            agency_name=agency_name)
        # Check that new agencies were appended
        self.assertEqual(keywords_agency['freedom'], ['Agency 1', 'Agency 2'])
        # Check that new words were added
        self.assertTrue('united' in keywords_agency)

    def test_get_tf_idf(self):
        """ Verifty that tfidf is calculated correctly """

        # Setup
        keywords = {
            'united': ['Agency 1', 'Agency 2', 'Agency 3'],
            'freedom': ['Agency 1', 'Agency 2'],
            'states': ['Agency 1']
        }
        agencies = {
            'Agency 1': ['united', 'freedom', 'states'],
            'Agency 2': ['united', 'freedom'],
            'Agency 3': ['united']
        }
        total_agencies = 3

        # Testing a words that all agencies have
        keyword = 'united'
        agency = 'Agency 1'
        score_united = clean_keywords.get_tf_idf(
            keyword=keyword,
            keywords=keywords,
            agency=agency,
            agencies=agencies,
            total_agencies=total_agencies)
        self.assertEqual(score_united, 0.0)

        # Testing with a word that only 2 agencies have
        keyword = 'freedom'
        agency = 'Agency 1'
        score_freedom = clean_keywords.get_tf_idf(
            keyword=keyword,
            keywords=keywords,
            agency=agency,
            agencies=agencies,
            total_agencies=total_agencies)
        self.assertTrue(score_freedom > score_united)

        # Testing with a word that only 1 agency has
        keyword = 'states'
        agency = 'Agency 1'
        score_states = clean_keywords.get_tf_idf(
            keyword=keyword,
            keywords=keywords,
            agency=agency,
            agencies=agencies,
            total_agencies=total_agencies)
        self.assertTrue(score_states > score_freedom)

    def test_clean_keywords(self):
        """
        Verify that keywords with scores lower than the threshold are removed
        """

        keywords = ['freedom', 'united', 'states', 'information']
        scores = [0, 0, 1, 4]
        new_keywords = clean_keywords.clean_keywords(keywords, scores)
        self.assertEqual(new_keywords, ['information', 'states'])

        # Wont' break if there is only one keyword
        keywords = ['freedom']
        scores = [1]
        new_keywords = clean_keywords.clean_keywords(keywords, scores)
        self.assertEqual(new_keywords, keywords)

    def test_extract_agency_keywords(self):
        """
        Verify that keywords are transformed into a dict that maps keywords
        to agencies and a dict that transforms agencies to keywords
        """

        keywords_agency = {}
        agency_keywords = {}
        agency = {'name': 'Agency 1', 'departments': []}

        # Test agency with no keywords
        clean_keywords.extract_agency_keywords(
            agency=agency,
            agency_keywords=agency_keywords,
            keywords_agency=keywords_agency)

        self.assertEqual(keywords_agency, {})
        self.assertEqual(agency_keywords, {})

        # Test agency with keywords
        agency['keywords'] = ['freedom', 'information', 'act']
        clean_keywords.extract_agency_keywords(
            agency=agency,
            agency_keywords=agency_keywords,
            keywords_agency=keywords_agency)

        self.assertEqual(
            keywords_agency,
            {
                'freedom': ['Agency 1'],
                'information': ['Agency 1'],
                'act': ['Agency 1']
            })
        self.assertEqual(
            agency_keywords, {'Agency 1': ['freedom', 'information', 'act']})

        # Test adding another agency that is in departments keywords
        agency = {'name': 'Agency 2', 'departments': []}
        agency['departments'] = [
            {'name': 'Office 1', 'keywords': ['freedom', 'united']}]
        clean_keywords.extract_agency_keywords(
            agency=agency,
            agency_keywords=agency_keywords,
            keywords_agency=keywords_agency)
        self.assertTrue('Office 1' in keywords_agency['freedom'])
        self.assertTrue('Office 1' in keywords_agency['united'])
        self.assertTrue('united' in agency_keywords['Office 1'])
        self.assertTrue('freedom' in agency_keywords['Office 1'])

    def test_apply_tf_idf(self):

        agency = {
            'name': 'Agency 1',
            'keywords': ['united', 'freedom', 'states'],
            'departments': [{
                'name': 'Office 1',
                'keywords': ['freedom', 'united', 'information']
            }]
        }
        # Setup
        keywords = {
            'united': ['Agency 1', 'Agency 2', 'Agency 3', 'Office 1'],
            'freedom': ['Agency 1', 'Agency 2', 'Office 1'],
            'states': ['Agency 1'],
            'information': ['Office 1']
        }
        agencies = {
            'Agency 1': ['united', 'freedom', 'states'],
            'Agency 2': ['united', 'freedom'],
            'Agency 3': ['united'],
            'Office 1': ['freedom', 'united', 'information']
        }
        total_agencies = 4

        data = clean_keywords.apply_tf_idf(
            agency=agency,
            agencies=agencies,
            keywords=keywords,
            total_agencies=total_agencies
        )

        self.assertEqual(data['keywords'], ['states'])
        self.assertEqual(data['departments'][0]['keywords'], ['information'])
