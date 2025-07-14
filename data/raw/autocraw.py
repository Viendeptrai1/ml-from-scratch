from icrawler.builtin import GoogleImageCrawler
# chó, mèo, sư tử, gà
animals = ['dogs', 'cats', 'tigers', 'chickens']

for animal in animals:
    crawler = GoogleImageCrawler(storage={'root_dir': f'./data/raw/{animal}'})
    crawler.crawl(keyword=animal, max_num=5, filters={'size': 'medium'})