from __future__ import unicode_literals # To handle non-english characters a la python 3
import requests
from bs4 import BeautifulSoup
import unicodecsv as csv
import os
import errno
import logging

'''
Table Format on ATP website
'''
WIN_SEED_IDX = 0
WIN_IDX = 2
LOSE_SEED_IDX = 4
LOSE_IDX = 6
SCORE_IDX = 7

PARSER = "lxml"


def process_table_row(row_soup, name, year, tourney_type, code, readable_name):
    if row_soup is None:
        win_seed = ""
        winner = ""
        lose_seed = ""
        loser = ""
        score = ""
    else:
        cols = [col.get_text().strip() for col in row_soup.find_all("td") if col.get_text()]
        win_seed = cols[WIN_SEED_IDX][1:-1]  # Removes parentheses
        winner = cols[WIN_IDX]
        lose_seed = cols[LOSE_SEED_IDX][1:-1]  # Removes parentheses
        loser = cols[LOSE_IDX]
        score = cols[SCORE_IDX]
    return win_seed, winner, lose_seed, loser, score, name, code, readable_name, year, tourney_type


def get_tournaments(archive):
    # tournaments = archive.find_all("ul", {'id': "tournamentDropdown"})
    # assert len(tournaments) == 1
    # return tournaments[0]
    tournaments_list = []  # List of tuples (name, code)
    tournaments_table = archive.find("table")
    rows = [row for row in tournaments_table.find_all("tr")]
    for row in rows:
        cols = [col for col in row.find_all("td") if col.get_text()]
        if len(cols) > 0:
            readable_name = row.find("span").text.strip()
            try:
                link = cols[-1].find("a").attrs["href"].split("/")
                name = link[4]
                code = link[5]
                tournaments_list.append((name, code, readable_name))
            except KeyError:
                logging.error("{} does not have results".format(readable_name))
    return tournaments_list


def process_tournament(year, name, code, tourney_type, readable_name):
    out = []
    result_endpoint = "http://www.atpworldtour.com/en/scores/archive/{}/{}/{}/results?matchType=singles".format(name, code, year)
    result_page = BeautifulSoup(requests.get(result_endpoint).text, PARSER)
    try:
        results = result_page.find("div", {'id': 'scoresResultsContent'})
        results_table = results.find("table")
        rows = [row for row in results_table.find_all("tr")]
        for row in rows:
            if row.find("td") is not None:
                out.append(process_table_row(row, name, year, tourney_type, code, readable_name))
        return out
    except AttributeError:
        logging.error("No results found for the {} {} when processing {}".format(year, name, tourney_type))
        return [process_table_row(None, name, year, tourney_type, code, readable_name)]


def ensure_dir_exists(tourney_type):
    path = "data/{}/".format(tourney_type)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_path(year, tourney_type, num=0, copy_flag=True):
    if num != 0:
        tentative_path = "data/{}/{}_{}.csv".format(tourney_type, year, num)
    else:
        tentative_path = "data/{}/{}.csv".format(tourney_type, year)
    if not os.path.isfile(tentative_path):
        return tentative_path
    if copy_flag:
        return get_path(year, tourney_type, num=num+1)
    return None


def create_csv_year(year, tourney_type, copy_flag=True):

    if tourney_type == "futures":
        type_code = "fu"
    elif tourney_type == "challenger":
        type_code = "ch"
    elif tourney_type == "world":
        type_code = "atpgs"
    elif tourney_type == "gs":
        type_code = "gs"
    else:
        raise ValueError("Unrecognized Tournament Type")

    ensure_dir_exists(tourney_type)
    f_path = get_path(year, tourney_type, copy_flag=copy_flag)
    if f_path is None:
        logging.info("Skipping {} {} due to user input".format(year, tourney_type))
    else:
        endpoint = "http://www.atpworldtour.com/en/scores/results-archive?year={}&tournamentType={}".format(year, type_code)
        archive = BeautifulSoup(requests.get(endpoint).text, PARSER)
        tournaments = get_tournaments(archive)

        data = []

        for name, code, readable_name in tournaments:
            print "Processing {}: {} ({})".format(year, name, readable_name)
            logging.info("\t{} {} ({})".format(year, name, readable_name))
            data.extend(process_tournament(year, name, code, tourney_type, readable_name))

        with open(f_path, 'wb+') as f:
            data_writer = csv.writer(f, delimiter=','.encode('utf-8'))
            header = ["Winner Seed",
                      "Winner",
                      "Loser Seed",
                      "Loser",
                      "Score",
                      "Tournament",
                      "Tournament Code",
                      "Tournament Pretty Name",
                      "Year",
                      "Type"]
            data_writer.writerow(header)
            for record in data:
                data_writer.writerow(record)


def main(copy_flag=True):
    logging.basicConfig(filename='log.log', level=logging.INFO)
    logging.info('Started')
    tourneys = ["world", "challenger", "futures"]
    years = range(1950, 2015)

    for tourney in tourneys:
        for year in years:
            logging.info("Processing {} Tournaments from {}".format(tourney, year))
            create_csv_year(year, tourney, copy_flag=copy_flag)
            logging.info("Finished {} Tournaments from {}".format(tourney, year))


if __name__ == '__main__':
    make_copy_choice = raw_input("Make copy of existing? (Y/n)").lower()
    if make_copy_choice == "" or make_copy_choice == "y":
        make_copy = True
    elif make_copy_choice == "n":
        make_copy = False
    else:
        while make_copy_choice not in ["", "y", "n"]:
            make_copy_choice = raw_input("Invalid Input. Make copy of existing? (Y/n)").lower()

    main(make_copy)
