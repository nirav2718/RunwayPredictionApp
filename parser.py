import re
import numpy as np
from datetime import datetime


class TimestampParser:
    def __init__(self):
        pass

    def timestamp_parser(self, text):
        obj = datetime.strptime(text, "%Y-%m-%d %H:%M")
        month = obj.month
        hour = obj.hour
        week_of_year = obj.isocalendar()[1]
        day_cos = np.cos(hour * (2 * np.pi / 24))
        day_sin = np.sin(hour * (2 * np.pi / 24))
        month_cos = np.cos(month * (2 * np.pi / 12))
        month_sin = np.cos(month * (2 * np.pi / 12))
        return {"week_of_year": week_of_year,
                "day_cos": day_cos, "day_sin": day_sin, "month_cos": month_cos,
                "month_sin": month_sin}


class MetarParser:
    def __init__(self):
        pass

    def findVisibility(self, text):
        pattern = r"\s\d{4}\s"
        pattern = re.compile(pattern)
        vis = re.search(pattern, text)
        if vis:
            span = vis.span()
            vis = int(text[span[0] + 1:span[1] - 1])
        return vis

    def findPressure(self, text):
        pattern = r"\sQ\d{4}\s"
        pattern = re.compile(pattern)
        pressure = re.search(pattern, text)
        if pressure:
            span = pressure.span()
            pressure = int(text[span[0] + 2:span[1] - 1])
        return pressure

    def findTempAndDew(self, text):
        pattern = r"\s((\d{2})|(M\d{2}))/((\d{2})|(M\d{2}))\s"
        pattern = re.compile(pattern)
        temp = re.search(pattern, text)
        if temp:
            temp = temp.group(1)
            if len(temp) == 3:
                temp = -int(temp[1:])
            else:
                temp = int(temp)
        dew = re.search(pattern, text)
        if dew:
            dew = dew.group(4)
            if len(dew) == 3:
                dew = -int(dew[1:])
            else:
                dew = int(dew)
        return temp, dew

    def findWind(self, text):
        wind_direction = None
        wind_speed = None
        pattern = r"\s(\d{3}|VRB)\d{2}(G\d\d)*KT\s"
        pattern = re.compile(pattern)
        wind = re.search(pattern, text)
        if wind:
            span = wind.span()
            wind_direction = text[span[0] + 1: span[0] + 4]
            wind_speed = int(text[span[0] + 4: span[0] + 6])
        return wind_speed, wind_direction

    def findCloud(self, text):
        cloud = None
        cloud_height = None
        pattern = r"(\sNSC\s)|([A-Z]{3}\d{3})"
        pattern = re.compile(pattern)
        cloud = re.search(pattern, text)
        if cloud:
            cloud = cloud.group().strip()
            if cloud != "NSC":
                cloud_height = int(cloud[-3:])
                cloud = cloud[:3]
            else:
                # in case of NSC, no cloud below 5000ft
                cloud_height = 50
        return cloud, cloud_height

    def findWeather(self, text):
        pattern = r"(BR|DS|DU|DZ|FC|FG|FU|GR|GS|HZ|IC|PE|PO|PY|RA|SA|SG|SN|SQ|SS|UP|VA)"
        pattern = re.compile(pattern)
        weather = re.search(pattern, text)
        if weather:
            span = weather.span()
            weather = text[span[0]: span[1]]
        return weather

    def isCavok(self, text):
        pattern = "\sCAVOK\s"
        pattern = re.compile(pattern)
        cavok = re.search(pattern, text)
        if cavok:
            return True
        return False

    def parse(self, text):
        if len(text.split()[0]) != 4 or len(text.split()[1]) != 7:
            return -1;

        # feature_list = text.split()
        airport = text.split()[0]
        # print("airport:", airport)
        time_zulu = text.split()[1]
        # print("time:", time_zulu)
        vis = self.findVisibility(text)
        # print("visibility:", vis)
        pressure = self.findPressure(text)
        # print("pressure:", pressure)
        temp, dew = self.findTempAndDew(text)
        # print("temp:", temp)
        # print("dew:", dew)
        wind_speed, wind_direction = self.findWind(text)
        # print("wind speed:", wind_speed)
        # print("wind direction:", wind_direction)
        cloud, cloud_height = self.findCloud(text)
        # print("cloud:", cloud)
        # print("cloud height:", cloud_height)
        weather = self.findWeather(text)
        # print("weather:", weather)
        cavok = self.isCavok(text)
        # print("cavok:", cavok)
        return {"time": time_zulu, "airport": airport, "wind_speed": wind_speed,
                "wind_direction": wind_direction, "visibility": vis, "weather": weather,
                "cloud": cloud, "cloud_height": cloud_height, "temprature": temp,
                "dew_point": dew, "pressure": pressure, "cavok": cavok}


# p = MetarParser()
# text = "VIDP 031700Z 001003G03KT 2000 R28/P2000 R29/P2000 CAVOK RAGR NSC Q1017 NOSIG"
# print(text)
# p.parse(text)
