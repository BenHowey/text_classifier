import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to load model 
@st.cache_resource
def load_model():
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"  
    return pipeline("zero-shot-classification", model=model_name)

def classify_text(texts, labels):
    classifier = load_model()
    results = classifier(texts, candidate_labels=labels, multi_label=True)
    return results

def main():
 
    # Title and description
    st.markdown("<h1 style='text-align: center; color: #002663;'>Text Classification App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #F28B00; '>Upload a CSV, input labels, and classify text using AI.</p>", unsafe_allow_html=True)

    # Upload CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows
        st.markdown("### Preview of Uploaded File:")
        st.write(df.head())

        # Ensure the 'text' exists
        if 'text' in df.columns:
            # Get labels from the user
            labels = st.text_input("Enter labels (comma-separated)", "")
            labels = [label.strip() for label in labels.split(",")]

            if labels:
                # Set a threshold for label selection
                threshold = st.slider("Select a threshold for label prediction", 0.0, 1.0, 0.95)

                # Run the classification
                with st.spinner("Classifying..."):
                    results = classify_text(df['text'].tolist(), labels)

                # Create a new column for the predicted labels
                df['predicted_labels'] = [
                    ", ".join([result['labels'][i] for i, score in enumerate(result['scores']) if score >= threshold])
                    for result in results
                ]

                # Also include individual label scores
                for label in labels:
                    df[f'{label}_score'] = [result['scores'][result['labels'].index(label)] for result in results]

                # Display the updated DataFrame
                st.markdown("### Classified Data:")
                st.write(df[['text', 'predicted_labels'] + [f'{label}_score' for label in labels]].head())

                # Visualize the count of each predicted label
                st.markdown("### Count of Predicted Labels:")
                label_counts = pd.Series([label for labels in df['predicted_labels'] for label in labels.split(", ") if label]).value_counts()
                st.bar_chart(label_counts)

                # Download the CSV
                csv = df.to_csv(index=False)
                st.download_button(label="Download classified data", data=csv, file_name='classified_data.csv', mime='text/csv')
        else:
            st.error("The uploaded file must contain a 'text' column.")

    # Footer or additional notes
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Built with ❤️ by RNLI Data Science using Streamlit and Hugging Face 🤗</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()




