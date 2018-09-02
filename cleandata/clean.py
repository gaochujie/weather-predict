from getdata.insertdb import selectarray
from getdata.insertdb import insertall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#获取表
def gettable(sql):
    s = selectarray(sql)
    table = pd.DataFrame(s, columns=["wdate","hightemper", "lowtemper",
                                     "iconstate", "humidity", "cloudCover", "windSpeed"])
    return table
#数据获取
def get_data(table):
    data_x = np.array([table["hightemper"], table["lowtemper"],
                     table["iconstate"], table["humidity"],
                     table["windSpeed"]],dtype=np.float32)
    data_y = np.array([table["cloudCover"]],dtype=np.float32)
    data_x = data_x.T
    data_y = data_y.T
    return data_x,data_y
# 计算最佳拟合曲线
def standRegress(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # linalg.det(xTx) 计算行列式的值
        print("This matrix is singular , cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws
#计算相关系数
def cleancloudCover(province,city):
    cloudCover = "cloudCover"
    sql = "SELECT wdate,hightemper,lowtemper,iconstate,humidity,cloudCover,windSpeed \
            FROM weatherdata WHERE province = '%s' AND city = '%s' AND \
            cloudCover<>999.98999 ORDER BY wdate ASC" % (province, city)
    table = gettable(sql)
    data_x,data_y = get_data(table)
    ws = standRegress(data_x, data_y)
    print("ws（相关系数）：", ws)  # ws 存放的就是回归系数
    insertsql = "INSERT INTO cleandata(province,city,factor,hightemper,\
    lowtemper,iconstate,humidity,windspeed) VALUES ('%s', '%s','%s',%f,%f,%f,%f,%f );" % \
                (province, city,cloudCover,ws[0],ws[1],ws[2],ws[3],ws[4])
    insertall(insertsql)
    #yHat = np.mat(data_x) * ws
    #plt.figure()
    #plt.plot(yHat-data_y, 'red')
    #plt.show()
#清洗数据，填补数据空缺
def updatecloudCover(province,city):
    cloudCover = "cloudCover"
    sql = "SELECT wdate,hightemper,lowtemper,iconstate,humidity,cloudCover,windSpeed \
                FROM weatherdata WHERE province = '%s' AND city = '%s' AND cloudCover=999.98999 \
                ORDER BY wdate ASC" % (province, city)
    xishusql = "SELECT province,city,factor,hightemper,lowtemper,iconstate,humidity,windspeed \
                FROM cleandata WHERE province = '%s' AND city = '%s' AND factor='%s'" %\
               (province, city,cloudCover)
    ws = selectarray(xishusql)
    ws=np.reshape(ws,[-1])
    ws = np.array(ws[3:8],dtype=np.float32)
    ws = np.mat(ws).T
    table = gettable(sql)
    datetable = table["wdate"]
    data_x, data_y = get_data(table)
    print(datetable)
    yHat = np.mat(data_x) * ws
    for i in range(len(yHat)):
        updatesql = "UPDATE weatherdata SET cloudCover = %f \
         WHERE province = '%s' AND city = '%s' AND wdate = '%s';"%(yHat[i],province,city,datetable[i])
        insertall(updatesql)
if __name__=='__main__':
    province = "上海市"
    city = "上海市"
    cleancloudCover(province,city)
    updatecloudCover(province,city)
