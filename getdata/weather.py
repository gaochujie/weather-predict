# -*- coding: utf-8 -*-

from weatherdata import *
from getdata.insertdb import selectdata,selectone,insertall
from others.getkey import getapikey
import time
from datetime import datetime, timedelta

def changetime(chtime):
    format = '%Y-%m-%d'
    value = time.localtime(chtime)
    dt = time.strftime(format, value)
    return dt

def changeicon(icon):
    sql="SELECT wnumber FROM weatherstate WHERE weatherico = '%s'" %icon
    num=selectone(sql)
    return num

def getweather(province,city):
    apikey = getapikey()
    sql="SELECT jingdu,weidu from citydata WHERE province='%s' AND city='%s'" %(province,city)
    lon,lat=selectdata(sql)
    for i in range(500):
        format = '%Y-%m-%d'
        value = time.time()
        dtime = value - 86400 * i
        dt = time.strftime(format, time.localtime(float(dtime)))
        strtime=dt+"T23:00:00"
        #print(strtime)
        fio = ForecastIO.ForecastIO(apikey,
                                    extend=(ForecastIO.ForecastIO.EXCLUDE_DAILY,),
                                    exclude=(ForecastIO.ForecastIO.EXCLUDE_CURRENTLY,
                                             ForecastIO.ForecastIO.EXCLUDE_HOURLY,
                                             ForecastIO.ForecastIO.EXCLUDE_MINUTELY,
                                             ForecastIO.ForecastIO.EXCLUDE_FLAGS),
                                    units=ForecastIO.ForecastIO.UNITS_SI,
                                    lang=ForecastIO.ForecastIO.LANG_CHINESE,
                                    latitude=lat, longitude=lon,
                                    time_url=strtime)
        #print('Latitude', fio.latitude, 'Longitude', fio.longitude)
        #print('Timezone', fio.timezone, 'Offset', fio.offset)
        print(fio.get_url())  # You might want to see the request url
        if fio.has_daily() is True:
            daily = FIODaily.FIODaily(fio)
            #print('Daily')
            #for item in daily.get_day(0).keys():
            #    print(item + ' : ' + str(daily.get_day(0)[item]))
            try:
                # 最高温
                temperatureHigh = daily.get_day(0)["temperatureHigh"]
            except:
                temperatureHigh = 999.99

            try:
                # 最低温
                temperatureLow = daily.get_day(0)["temperatureLow"]
            except:
                temperatureLow = 999.99

            try:
                #湿度
                humidity = daily.get_day(0)["humidity"]
            except:
                humidity = 999.99

            try:
                #总体情况
                icon = daily.get_day(0)["icon"]
                # 描述转数据
                ico = changeicon(icon)
            except:
                ico=0

            try:
                #最大降雨强度
                precipIntensityMax = daily.get_day(0)["precipIntensityMax"]
            except:
                precipIntensityMax = 999.99

            try:
                #日期
                weatherdate = daily.get_day(0)["time"]
                # 转换时间
                wtime = changetime(weatherdate)
            except:
                wtime="9999-99-99"

            try:
                # 降雨概率
                precipProbability = daily.get_day(0)["precipProbability"]
            except:
                precipProbability = 999.99

            try:
                #云图
                cloudCover = daily.get_day(0)["cloudCover"]
            except:
                cloudCover = 999.99

            try:
                #风速
                windSpeed = daily.get_day(0)["windSpeed"]
            except:
                windSpeed = 999.99
            insertsql="INSERT INTO weatherdata\
            (wdate,city,province,hightemper,lowtemper,iconstate,humidity,\
            precipProbability,precipIntensityMax,cloudCover,windSpeed) \
            VALUES ('%s', '%s', '%s', %f, %f, %d, %f, %f, %f, %f, %f );" %\
                      (wtime, city, province, temperatureHigh, temperatureLow, ico,
                       humidity,precipProbability,precipIntensityMax,cloudCover,windSpeed)
            insertall(insertsql)
        else:
            print('No Daily data')
#getweather("广东省","广州市")
