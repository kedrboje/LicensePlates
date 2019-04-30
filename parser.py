import requests
from bs4 import BeautifulSoup
 

class img_parser():

    def __init__(self, url):
        self.url = url
        try:
            self.page = requests.get(url)
            self.soup = BeautifulSoup(self.page.text, 'html.parser')
        except:
            print("Url Error")

    def get_links_storage(self, attr: str, links_storage_path: str):
        self.storage = self.soup.findAll('img')  # here would be your tag
        with open(links_storage_path, 'a') as f:
            for link in self.storage[1:]:  # delete [1:] if you need first item
                f.write(str(link.get(attr)) + "\n")

    def get_img_storage(self, links_storage_path: str, img_storage_path: str, extension=".jpg"):
        with open(links_storage_path, 'r') as f:
            for index, line in enumerate(f):
                try:
                    with open(img_storage_path + str(index) + extension, 'wb') as f_img:
                        f_img.write(requests.get(line).content)
                except:
                    pass

