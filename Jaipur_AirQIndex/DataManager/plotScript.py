import pandas as pd 
import matplotlib.pyplot as plt


def avg_Data(year):
    temp_i = 0
    average = []
    for rows in pd.read_csv('DataCSV/aqi{}.csv'.format(year), chunksize=24):
        addVar = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data = rows)
        for index,row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data: 
            if type(i) is float or type(i) is int:
                addVar = addVar + i
            elif type(i) is str:
                if i!= 'NoData' and i!= 'PwrFail' and i!= '---' and i!= 'InVld':
                    temp = float(i)
                    addVar = addVar + temp
        avg = addVar/24
        temp_i = temp_i+1
        average.append(avg)
    return average



if __name__=="__main__":
    newList = []

    for year in range(2013,2019):
        a = avg_Data(year)
        newList.append(a)
    print(len(newList))
    for j in range(0,len(newList)):
        print(len(newList[j]))
        break

    #print(a)

    # lst2013=avg_data_2013()
    # lst2014=avg_data_2014()
    # lst2015=avg_data_2015()
    # lst2016=avg_data_2016()s
    # lst2017=avg_data_2017()
    # lst2018=avg_data_2018()
    plt.plot(a,label="2013 data")
    # plt.plot(range(0,364),lst2014,label="2014 data")
    # plt.plot(range(0,365),lst2015,label="2015 data")
    # plt.plot(range(0,121),lst2016,label="2016 data")
    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.legend(loc='upper right')
    plt.show()