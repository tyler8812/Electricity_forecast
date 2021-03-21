import pandas as pd
 
past = pd.read_csv("./台灣電力公司_過去電力供需資訊.csv", usecols=["備轉容量(MW)"])
print(past)
now = pd.read_csv("./本年度每日尖峰備轉容量率.csv", usecols=[1])
print(now.values*10)
print(type(now))