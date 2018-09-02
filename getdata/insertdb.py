#coding:utf-8

import pymysql
import json
import numpy as np


def to_mysql():
    with open('..\\jsondir\\info.json') as credentials_file:
        credentials = json.load(credentials_file)
    host=credentials['host']
    user=credentials['user']
    passwd = credentials['password']
    db=credentials['database']
    conn = pymysql.connect(host=host,
                           user=user,
                           passwd=passwd,
                           db=db,
                           charset='utf8',
                           autocommit=True)
    return conn

def insertall(strsql):
  #  sql = getcity()
    sql=strsql
    conn = to_mysql()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
        print("执行成功")
    except Exception as e:
       conn.rollback()
       print(e.args,"执行失败")
       cur.close()
       conn.close()

def selectdata(strsql):
    sql=strsql
    conn=to_mysql()
    cur=conn.cursor()
    global lon
    global lat
    lon = lat = 0.0
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            lon = row[0]
            lat = row[1]
            print("lon=%s,lat=%s" %(lon, lat))
    except Exception as e:
        print(e.args)
    cur.close()
    conn.close()
    return lon,lat

def selectone(strsql):
    sql = strsql
    conn = to_mysql()
    cur = conn.cursor()
    global w_str
    w_str=0
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            w_str = row[0]
            #print(w_str)
    except Exception as e:
        print(e.args)
    cur.close()
    conn.close()
    return w_str

def selectarray(strsql):
    sql = strsql
    conn = to_mysql()
    cur = conn.cursor()
    global wstr
    try:
        cur.execute(sql)
        results = cur.fetchall()
        wstr = results
    except Exception as e:
        print(e.args)
    cur.close()
    conn.close()
    wstr = np.array(wstr)
    return wstr
