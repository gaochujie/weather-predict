from getdata.insertdb import selectone,selectdata
from getdata.weather import changeicon,changetime
from others.getkey import getapikey
from getdata.insertdb import insertall
from weatherdata import *
import time
import datetime


#获取当前时间
def getnowtime():
    format = '%Y-%m-%d'
    value = time.time()
    dt = time.strftime(format, time.localtime(float(value)))
    return dt

#获取时间序列
def gettimestep(i):
    value = time.time()
    dtime = value - 86400 * i
    dt = time.strftime(format, time.localtime(float(dtime)))
    strtime = dt + "T23:00:00"
    return strtime

#更新数据
def updateweather(province,city):
    apikey = getapikey()
    sql="SELECT MAX(wdate) FROM weatherdata \
    WHERE province = '%s' AND city = '%s' ORDER BY wdate DESC;"%(province,city)
    global format
    format = "%Y-%m-%d"
    lasttime=selectone(sql)
    lasttime = datetime.datetime.strptime(str(lasttime),format)
    nowtime = getnowtime()
    nowtime = datetime.datetime.strptime(nowtime,format)
    delta = (nowtime - lasttime).days
    if delta >0:
        ssql = "SELECT jingdu,weidu from citydata WHERE province='%s' AND city='%s'" % (province, city)
        lon, lat = selectdata(ssql)
        for i in range(1,delta):
            strtime = gettimestep(i)
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
            print(fio.get_url())
            if fio.has_daily() is True:
                daily = FIODaily.FIODaily(fio)
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
                    # 湿度
                    humidity = daily.get_day(0)["humidity"]
                except:
                    humidity = 999.99

                try:
                    # 总体情况
                    icon = daily.get_day(0)["icon"]
                    # 描述转数据
                    ico = changeicon(icon)
                except:
                    ico = 0

                try:
                    # 最大降雨强度
                    precipIntensityMax = daily.get_day(0)["precipIntensityMax"]
                except:
                    precipIntensityMax = 999.99

                try:
                    # 日期
                    weatherdate = daily.get_day(0)["time"]
                    # 转换时间
                    wtime = changetime(weatherdate)
                except:
                    wtime = "9999-99-99"

                try:
                    # 降雨概率
                    precipProbability = daily.get_day(0)["precipProbability"]
                except:
                    precipProbability = 999.99

                try:
                    # 云图
                    cloudCover = daily.get_day(0)["cloudCover"]
                except:
                    cloudCover = 999.99

                try:
                    # 风速
                    windSpeed = daily.get_day(0)["windSpeed"]
                except:
                    windSpeed = 999.99
                insertsql = "INSERT INTO weatherdata\
                        (wdate,city,province,hightemper,lowtemper,iconstate,humidity,\
                        precipProbability,precipIntensityMax,cloudCover,windSpeed) \
                        VALUES ('%s', '%s', '%s', %f, %f, %d, %f, %f, %f, %f, %f );" % \
                            (wtime, city, province, temperatureHigh, temperatureLow, ico,
                             humidity, precipProbability, precipIntensityMax, cloudCover, windSpeed)
                insertall(insertsql)
            else:
                print('No Daily data')
    else:
        print("数据已是最新，无需更新")

#updateweather("北京市","北京市")


