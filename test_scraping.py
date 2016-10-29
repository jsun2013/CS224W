import unittest
import scraping
from bs4 import BeautifulSoup
import requests


class TestWebsite(unittest.TestCase):
    def testGetTournaments(self):
        year = 2014
        type_code = "gs"
        endpoint = "http://www.atpworldtour.com/en/scores/results-archive?year={}&tournamentType={}".format(year, type_code)
        archive = BeautifulSoup(requests.get(endpoint).text, 'lxml')
        tournaments = scraping.get_tournaments(archive)
        self.assertEqual(len(tournaments), 4)

    def testGetPath(self):
        # Test if no file exists:
        year = 2014
        tourney_type = "_debug"
        fpath = scraping.get_path(year, tourney_type)
        self.assertEqual(fpath, "data/_debug/2014.csv")

        # test if file exists. There should be a /data/_debug/2015.csv file in the current directory
        year = 2015
        tourney_type = "_debug"
        fpath = scraping.get_path(year, tourney_type)
        self.assertEqual(fpath, "data/_debug/2015_1.csv")
