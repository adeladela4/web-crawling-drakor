import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Load the trained pipeline
pipeline = joblib.load('model/model_lr.joblib')


# =========================================================
# ANALYZE TOKEN SENTIMENT
# =========================================================
def analyze_token_sentiment(text, pipeline):
    tokens = text.split()
    token_sentiments = []

    for token in tokens:
        sentiment = pipeline.predict([token])[0]

        token_sentiments.append({
            "token": token,
            "sentiment": sentiment
        })

    return token_sentiments


# =========================================================
# GENERATE WORDCLOUD
# =========================================================
def generate_wordcloud(data):
    text = " ".join(data['full_text'].astype(str))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(
        wordcloud,
        interpolation='bilinear'
    )

    ax.axis("off")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# PLOT TOP WORDS
# =========================================================
def plot_top_words(data):
    text = " ".join(data['full_text'].astype(str))

    words = text.lower().split()

    word_counts = Counter(words)

    top_words = word_counts.most_common(10)

    words_df = pd.DataFrame(
        top_words,
        columns=['Word', 'Frequency']
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        words_df['Word'],
        words_df['Frequency']
    )

    ax.set_title("Top 10 Most Frequent Words")
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)


def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Project")

    # Add sidebar with menu options
    menu = ["Home", "Dataset & Analysis", "KMeans Clustering", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # =========================================================
    # HOME
    # =========================================================
    if choice == "Home":
        st.subheader("Home")

        with st.form("nlp_form"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

            # Layout
            col1, col2 = st.columns(2)

            if submit_button:
                with col1:
                    st.info("Result")

                    # Predict sentiment
                    sentiment = pipeline.predict([raw_text])[0]

                    st.write(f"Predicted Sentiment: {sentiment}")

                    # Emoji based on sentiment
                    if sentiment == "positive":
                        st.markdown("🟢 Positive 😊")

                    elif sentiment == "negative":
                        st.markdown("🔴 Negative 😢")

                    else:
                        st.markdown("🟡 Neutral 😐")

                with col2:
                    st.info("Token Sentiment")

                    # Analyze sentiment for individual tokens
                    token_sentiments = analyze_token_sentiment(
                        raw_text,
                        pipeline
                    )

                    st.write(token_sentiments)

    # =========================================================
    # DATASET & ANALYSIS
    # =========================================================
    elif choice == "Dataset & Analysis":
        st.subheader("Dataset & Sentiment Analysis")

        # Load dataset
        dataset_path = 'data/Drakor_predicted.csv'
        data = pd.read_csv(dataset_path)

        # Display dataset
        st.write("📊 Dataset")
        st.dataframe(data)

        # Count sentiment distribution
        sentiment_counts = data['sentiment'].value_counts()

        st.write("📈 Distribusi Sentimen")
        st.bar_chart(sentiment_counts)

        # Display number of sentiments
        st.write("🔢 Jumlah Sentiment")
        st.write(sentiment_counts)

        # Generate Word Cloud
        st.write("☁️ Word Cloud")
        generate_wordcloud(data)

        # Plot top frequent words
        st.write("📊 Most Frequent Words")
        plot_top_words(data)

        # Display dataset with sentiment labels
        st.write("📋 Dataset with Sentiment Labels")
        st.dataframe(
            data[['full_text', 'sentiment']]
        )

    # =========================================================
    # KMEANS CLUSTERING
    # =========================================================
    elif choice == "KMeans Clustering":
        st.subheader("KMeans Clustering Analysis")

        # Load dataset
        dataset_path = 'data/Drakor_predicted.csv'
        data = pd.read_csv(dataset_path)

        # Convert text data to list
        text_data = data['full_text'].to_list()

        # Vectorization
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text_data)

        # KMeans clustering
        true_k = 8

        model = KMeans(
            n_clusters=true_k,
            init='k-means++',
            max_iter=100,
            n_init=1,
            random_state=42
        )

        model.fit(X)

        # Display clusters
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names_out()

        cluster_info = ""

        for i in range(true_k):
            cluster_info += (
                f"Cluster {i}: "
                + " ".join(
                    [terms[ind] for ind in order_centroids[i, :10]]
                )
                + "\n"
            )

        st.write("📌 Cluster Terms")
        st.text(cluster_info)

        # Silhouette Score
        score = silhouette_score(X, model.labels_)

        st.write(
            f"📊 Silhouette Score: {score:.2f}"
        )

        # =====================================================
        # PCA FOR 2D VISUALIZATION
        # =====================================================

        pca = PCA(
            n_components=2,
            random_state=42
        )

        reduced_features = pca.fit_transform(
            X.toarray()
        )

        reduced_cluster_centers = pca.transform(
            model.cluster_centers_
        )

        # Plot clusters
        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=model.labels_,
            cmap='viridis',
            alpha=0.6
        )

        # Plot centroids
        plt.scatter(
            reduced_cluster_centers[:, 0],
            reduced_cluster_centers[:, 1],
            marker='x',
            s=150,
            c='red',
            label='Centroids'
        )

        plt.title(
            "KMeans Clustering Visualization"
        )

        plt.xlabel(
            "PCA Component 1"
        )

        plt.ylabel(
            "PCA Component 2"
        )

        plt.colorbar(scatter)

        plt.legend()

        st.pyplot(plt)

        # Close figure
        plt.close()

    # =========================================================
    # ABOUT
    # =========================================================
    else:
        st.subheader("About")

        st.write(
            "This is a simple sentiment analysis app "
            "built using Streamlit."
        )


# Run the application
if __name__ == "__main__":
    main()