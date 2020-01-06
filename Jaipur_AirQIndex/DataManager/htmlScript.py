import os
import time
import requests
import sys

#This function 
def get_html():
    for year in range(2013,2019):
        for month in range(1,13):
            if (month < 10): 
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-423480.html'.format(month,year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-423480.html'.format(month,year)

            texts = requests.get(url)
            textUTF = texts.text.encode('utf=8')

            if not os.path.exists("Data/Html_Data/{}".format(year)):
                os.makedirs("Data/Html_Data/{}".format(year))
            with open("Data/Html_Data/{}/{}.html".format(year,month),"wb") as output:
                output.write(textUTF)

        sys.stdout.flush()

if __name__ == "__main__":
    startTime = time.time()
    get_html()
    stopTime = time.time()
    print("Time Taken: {}".format(stopTime-startTime))