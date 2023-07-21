import numpy as np
import pickle
import streamlit as st
import pandas as pd
import base64
from PIL import Image

# loading the saved model
loaded_model = pickle.load(open("trained_model_sav", "rb"))

# creating function for prediction
def prediction(data):
    test_pred = loaded_model.predict(data)
    return test_pred

def main():
    # Insert the JPEG image using PIL and st.image
    image = Image.open("C:/Users/Rspse/Downloads/mechine learn/user_behaviour-1024x332-1.jpg")
    st.image(image, caption="User Behavior", use_column_width=True)

    # giving a title
    st.title('USER BEHAVIOR PREDICTION')

    # create a file uploader widget
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # read the uploaded CSV data
        user_data = pd.read_csv(uploaded_file)

        # display the uploaded data (optional)
        st.subheader("Uploaded Data")
        st.write(user_data)

        # code for prediction
        predictor = ''

        # create button
        if st.button('Cluster result'):
            predictor = prediction(user_data)

            # Display the cluster result for all data
            st.subheader("Cluster Result")
            st.write(predictor)

            # Attach the prediction as a new column
            user_data['Predicted'] = predictor

            # Save prediction to CSV file on the server
            user_data.to_csv("predicted_data.csv", index=False)

            # Provide a download link to the user
            st.markdown(get_download_link(user_data), unsafe_allow_html=True)

            # USER BEHAVIOR BASED ON CLUSTER
            st.title('USER BEHAVIOR BASED ON CLUSTER')
            behavior_df = pd.read_csv("C:/Users/Rspse/Downloads/mechine learn/avg.csv")
            st.dataframe(behavior_df)

            # REVENUE BASED ON CLUSTER
            st.title('REVENUE BASED ON CLUSTER')
            # Read the predicted data from the CSV file
            predicted_data = pd.read_csv("predicted_data.csv")

            # Calculate the total count of data points in each cluster
            cluster_counts = predicted_data['Predicted'].value_counts()

            # Group the data by cluster and calculate the percentage of 'true' values in the revenue column
            revenue_percentages = predicted_data.groupby('Predicted')['Revenue'].mean() * 100

            # Combine the cluster counts and revenue percentages into a single DataFrame
            result_df = pd.DataFrame({'Cluster Count': cluster_counts, 'Revenue Percentage': revenue_percentages})

            # Display the result
            st.dataframe(result_df)

def get_download_link(df):
    # Convert the DataFrame to CSV
    csv = df.to_csv(index=False)

    # Encode the CSV data as base64
    b64 = base64.b64encode(csv.encode()).decode()

    # Create a download link with HTML
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Download Prediction CSV</a>'
    return href

if __name__ == '__main__':
    main()
