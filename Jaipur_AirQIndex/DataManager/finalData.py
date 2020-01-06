from plotScript import avg_Data
import requests
import sys
import os,csv
import pandas as pd 
from bs4 import BeautifulSoup



def scrub_html_data(month,year):
    html_file = open('Data/Html_Data/{}/{}.html'.format(year,month),'rb')
    plain_text = html_file.read()

    temp_data = []
    final_data = []

    soup = BeautifulSoup(plain_text, "lxml")
    for table in soup.findAll('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody: 
                a = tr.get_text()
                temp_data.append(a)

    rows = len(temp_data)/15

    for times in range(round(rows)):
        newtempD = []
        for i in range(15):
            newtempD.append(temp_data[0])
            temp_data.pop(0)
        final_data.append(newtempD)  
        #print(final_data)
    length = len(final_data)      
    final_data.pop(length-1)
    final_data.pop(0)
    

    for a in range(len(final_data)):
        final_data[a].pop(4)
        final_data[a].pop(13)
        final_data[a].pop(12)
        final_data[a].pop(11)
        final_data[a].pop(10)
        final_data[a].pop(9)
        final_data[a].pop(0)

    return final_data


def mergeData(year,chunk):
    for a in pd.read_csv('Data/csvData/csv_' + str(year) + '.csv', chunksize=chunk):
        df = pd.DataFrame(data = a)
        mylist = df.values.tolist()
    return mylist



if __name__ == "__main__":
   if not os.path.exists("Data/csvData"):
       os.makedirs("Data/csvData")
   for year in range(2013,2019):
       fdata = []
       with open('Data/csvData/csv_'+str(year)+'.csv','w') as  csvfile:
           wr = csv.writer(csvfile, dialect='excel')
           wr.writerow(
               ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5']
               )
        
       for month in range(1,13): 
           temp = scrub_html_data(month,year)
           fdata = fdata+temp


       pm = avg_Data(year)

        
       for i in range(len(fdata)-1):
           fdata[i].insert(8,pm[i])

       with open('Data/csvData/csv_'+str(year)+'.csv','a') as  csvfile:
           wr = csv.writer(csvfile, dialect= 'excel')
           for j in fdata:
               k = 0
               for items in j:
                   if items == "" or items =="-":
                       k = 1
               if k!= 1: 
                   wr.writerow(j)
    
   csv2013 = mergeData(2013, 600)
   csv2014 = mergeData(2014, 600)
   csv2015 = mergeData(2015, 600)
   csv2016 = mergeData(2016, 600)
   csv2017 = mergeData(2017, 600)
   csv2018 = mergeData(2018, 600)
    

   total=csv2013+csv2014+csv2015+csv2016+csv2017+csv2018
   print(type(total))
    
   with open('Data/csvData/Final.csv', 'w') as csvfile:
       wr = csv.writer(csvfile, dialect='excel')
       wr.writerow(
           ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
       wr.writerows(total)
        
df = pd.read_csv('Data/csvData/Final.csv')