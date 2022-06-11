# # # # import re
# # # #
# # # # text = "VIDP 010030Z 00000KT 0200 R28/1200 R29/0800 FG FEW100 M06/M05 Q1019 BECMG 0150 FG"
# # # # pattern = r"\s((\d{2})|(M\d{2}))/((\d{2})|(M\d{2}))\s"
# # # # pattern = re.compile(pattern)
# # # # print(re.search(pattern, text).group(1))
# # # # print(re.search(pattern, text).group(4))
# # #
# # # from datetime import datetime
# # #
# # # obj = datetime.strptime("2020-03-08 00:00", "%Y-%m-%d %H:%M")
# # # print(obj.month)
# # # print(obj.hour)
# # # print(obj.isocalendar()[1])
# # import pandas as pd
# #
# # a = {"a": 1, "b": 2}
# # b = {"c": 3, "d": 4}
# # print(b)
# # b.pop("c")
# # print(b)
# import pandas as pd
#
# df = pd.read_csv("train_mean.csv", index_col="Unnamed: 0")
# df=df[df.columns[0]]
# print(df)
#
# print(type(df))

import pandas as pd

df = pd.read_csv("chennai_remaining_test_data.csv")

for row in zip(df.valid, df.metar):
    print(row)
    break