from lxml import html
import requests
import csv

NUM_COLUMNS = 1799
outwriter = csv.writer(open('./data/data_raw.csv', 'w'), quotechar='\"', delimiter=',')

page = requests.get('https://en.wikipedia.org/wiki/List_of_Emily_Dickinson_poems')
tree = html.fromstring(page.content)

num_valid_links = 0

for i in range(NUM_COLUMNS):
    try:
        poem_link = str(tree.xpath("//*[@id=\"mw-content-text\"]/div/table/tr[{}]/td[1]/a/@href".format(i+2))[0])
    except:
        # Poem has no link.
        continue

    poem_page = requests.get(poem_link)
    poem_tree = html.fromstring(poem_page.content)
    poem = poem_tree.xpath("//*[@id=\"mw-content-text\"]/div/div[2]/p/text()")

    if len(poem) > 0:
        outwriter.writerow(["".join(poem) + '\n'])
        num_valid_links += 1

print("{0} of {1} poems successfully scraped.".format(num_valid_links, NUM_COLUMNS))
