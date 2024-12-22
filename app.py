import streamlit as st
import pandas as pd
from nixtla import NixtlaClient


def update_data(data, column_to_delete=None, column_to_rename=None, new_column_name=None):
    if column_to_delete:
        data = data.drop(columns=[column_to_delete])
    if column_to_rename and new_column_name:
        data = data.rename(columns={column_to_rename: new_column_name})
    return data

def trainmodel(df,time, y):

    df[time] = pd.to_datetime(df[time])
    df.set_index(time, inplace=True)
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')
    df = df.reindex(full_time_range)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].interpolate(method='polynomial', order=3)
    df.reset_index(inplace=True)
    df.rename(columns={'index': time}, inplace=True)

    train_data = df.iloc[:-144]
    test_data = df.iloc[-144:]

    y_test = test_data[[time,y]]
    print(train_data.head(10))
    predict_df = nixtla_client.forecast(
            df=train_data,
            h=144,                            
            level=[90],                        
            model='timegpt-1-long-horizon',    
            time_col=time,
            target_col=y)
    
    return predict_df, y_test

def plotpred(test, pred, time, target):
    st.pyplot(nixtla_client.plot(
            test, pred, models=["TimeGPT"], level=[90], time_col=time, target_col=target))


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_icon="./logo.svg",
        page_title="EaSy Forecast",
    )
    st.title("EaSy Forecast")

    st.sidebar.header("Upload Your Data")
    input_file = st.sidebar.file_uploader(
        "Upload your data", type=["xlsx", "csv"], accept_multiple_files=False
    )
    
    api_key = st.text_input("Enter your API key", type="password")
    if api_key:
        nixtla_client = NixtlaClient(api_key = api_key)

    with st.container():

        if "data" not in st.session_state:
            st.session_state["data"] = None

        if input_file and st.session_state['data'] is None:
            if input_file.name.lower().endswith(".csv"):
                st.session_state['data'] = pd.read_csv(input_file)
            else:
                st.session_state['data'] = pd.read_excel(input_file)
        
        if st.session_state['data'] is not None:

            st.sidebar.subheader("Delete Column")
            column_to_delete = st.sidebar.selectbox("Select column to delete", st.session_state['data'].columns)
            delete_button = st.sidebar.button("Delete Column")

            st.sidebar.subheader("Rename Column")
            column_to_rename = st.sidebar.selectbox("Select a column to rename", st.session_state['data'].columns)
            new_column_name = st.sidebar.text_input("Enter new name for the column", "")
            rename_button = st.sidebar.button("Rename Column")

            if delete_button:
                st.session_state['data'] = update_data(st.session_state['data'], column_to_delete=column_to_delete)

            if new_column_name and rename_button:
                st.session_state['data'] = update_data(st.session_state['data'], column_to_rename=column_to_rename, new_column_name=new_column_name)

            st.dataframe(st.session_state['data'], use_container_width=True)

            st.subheader("Time")
            time_column = st.selectbox(
                "Select the Time column", st.session_state["data"].columns
            )
            st.subheader("Target Column")
            target_column = st.selectbox(
                "Select the column to predict", st.session_state["data"].columns)
            
            if st.button("Train Model"):
                if api_key:
                    pred, test = trainmodel(st.session_state['data'], time_column,target_column)
                    plotpred(test, pred, time_column, target_column)

        else:
            st.write("Upload your data")
