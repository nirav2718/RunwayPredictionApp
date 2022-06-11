import pandas as pd
import streamlit as st
from parser import MetarParser, TimestampParser
from datetime import datetime, timedelta

def cavokUpdate(features):
    if features["cavok"]:
        features["cloud_height"] = 50
        features["visibility"] = 10000
        features["weather"] = "NSW"
    features.pop("cavok")
    return features


def cloudDummy(df):
    known_cat = ['BKN', 'FEW', 'SCT']
    cats = pd.Categorical(df["cloud"], categories=known_cat)
    return pd.get_dummies(cats)


def weatherDummy(df):
    known_cat = ['BR', 'DZ', 'FG', 'FU', 'HZ', 'RA', 'TS']
    cats = pd.Categorical(df["weather"], categories=known_cat)
    return pd.get_dummies(cats)


st.sidebar.title("Runway Prediction App")

metar_text = st.text_input('Enter Metar text:')
timestamp = st.text_input("Enter timestamp:")
# st.write("Metar text is: ", metar_text)
# st.button("Predict")
if st.button("Predict"):
    metar_parser = MetarParser()
    tm_parser = TimestampParser()
    timestamp_features = tm_parser.timestamp_parser(timestamp)
    features = metar_parser.parse(metar_text)

    if features["wind_direction"] == "VRB":
        features["wind_direction"] = None
    if features["wind_direction"] == "360":
        features["wind_direction"] = "0"

    features = cavokUpdate(features)

    features.update(timestamp_features)
    df = pd.DataFrame(features, index=[0])

    # df = df.astype({"wind_direction": int})

    cloud_dummy = cloudDummy(df)
    weather_dummy = weatherDummy(df)

    # st.dataframe(df)
    # st.write("cloud")
    # st.dataframe(cloud_dummy)
    # st.write("weather")
    # st.dataframe(weather_dummy)

    df = pd.concat([df, cloud_dummy, weather_dummy], axis=1)
    # st.dataframe(df)

    df = df.drop(["time", "airport", "cloud", "weather"], axis=1)

    prod_data = pd.read_csv("production_47_data.csv")
    prod_data = prod_data.drop("Unnamed: 0", axis = 1)
    renamed_cols = {"BKN": "cloud_BKN", "FEW": "cloud_FEW", "SCT": "cloud_SCT",
                    "BR": "weather_BR", "DZ": "weather_DZ", "FG": "weather_FG",
                    "FU": "weather_FU", "HZ": "weather_HZ", "RA": "weather_RA",
                    "TS": "weather_TS"}
    df = df.rename(renamed_cols, axis=1)

    # st.dataframe(df)
    # st.dataframe(prod_data)
    # st.write(df.info())
    df = pd.concat([prod_data, df])

    # filling missing data
    df = df.fillna(method="ffill")

    df = df.astype({"wind_direction": int})

    # scaling

    # train mean
    train_mean = pd.read_csv("train_mean.csv", index_col="Unnamed: 0")
    train_mean = train_mean[train_mean.columns[0]]
    # train std
    train_std = pd.read_csv("train_std.csv", index_col="Unnamed: 0")
    train_std = train_std[train_std.columns[0]]
    # scaling
    scaled_df = (df - train_mean) / train_std
    # st.dataframe(scaled_df)

    # ready to feed numpy array
    np_array_to_predict = scaled_df.values

    np_array_to_predict = np_array_to_predict.reshape((1, 48, 22))
    # print(np_array_to_predict.shape)

    #load model wind speed
    from tensorflow import keras
    model = keras.models.load_model('wind_speed_model.h5')

    #predict wind speed
    pred = model.predict(np_array_to_predict)
    pred = pred * train_std['wind_speed'] + train_mean['wind_speed']
    # st.write(pred)



    # wind direction loading and prediction
    dir_model = keras.models.load_model('wind_direction_model_1.h5')
    wind_dir_pred = dir_model.predict(np_array_to_predict)
    wind_dir_pred = wind_dir_pred * train_std['wind_direction'] + train_mean['wind_direction']
    # st.write(wind_dir_pred)


    # print("-"*50)
    # print(pred.shape)
    # print("-"*50)

    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")


    col1, col2, col3 = st.columns(3)



    with col1:
        st.subheader("Time")
        for i in range(1, 7):
            st.write(((timestamp + timedelta(minutes=i * 30)).strftime("%H:%M")))

    with col2:
        st.subheader("Wind Speed")
        for i in range(1, 7):
            st.write(str(pred[0][i - 1]))

    with col3:
        st.subheader("Wind Direction")
        for i in range(1, 7):
            st.write(str(wind_dir_pred[0][i - 1]))

    # dropping 0th row and appending the most recent row
    # df = df.drop(index = 0, axis = 0).reset_index(drop = True)
    # df.to_csv("production_47_data.csv")


    # st.write("time:", features["time"])
    # st.write("airport:", features["airport"])
    # st.write("wind_speed:", features["wind_speed"])
    # st.write("wind_direction:", features["wind_direction"])
    # st.write("visibility:", features["visibility"])
    # st.write("weather:", features["weather"])
    # st.write("cloud:", features["cloud"])
    # st.write("cloud_height:", features["cloud_height"])
    # st.write("temprature:", features["temprature"])
    # st.write("dew:", features["dew_point"])
    # st.write("pressure:", features["pressure"])
    # st.write("cavok:", features["cavok"])

    # print(features)
