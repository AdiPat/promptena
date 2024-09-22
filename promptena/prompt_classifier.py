import pickle
import joblib
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class PromptContextClassifier:
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.vectorizer = None
        self.model = self._initialize_model(model_type)

        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(max_features=1000)

    def _initialize_model(self, model_type):
        if model_type == "logistic_regression":
            return LogisticRegression()
        elif model_type == "svm":
            return SVC()
        elif model_type == "decision_tree":
            return DecisionTreeClassifier()
        elif model_type == "dnn":
            return self._build_dnn()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _build_dnn(self, X_train=None):
        if X_train is None:
            return None

        tfidf_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=1000, output_mode="tf-idf", ngrams=(2, 3)
        )

        self.vectorizer = tfidf_vectorizer

        self.vectorizer.adapt(X_train)

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(
                    shape=(1,), dtype=tf.string
                ),  # Input layer for raw text
                tfidf_vectorizer,  # TextVectorization layer
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, vectorizer=TfidfVectorizer(max_features=1000)):
        if vectorizer:
            self.vectorizer = vectorizer
        X_train_vec = self.vectorizer.fit_transform(X_train)
        if self.model_type == "dnn":
            if self.model is None:
                self.model = self._build_dnn(X_train)
            self.model.fit(
                X_train,
                y_train,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
            )
        else:
            self.model.fit(X_train_vec, y_train)

    def predict(self, X_test):
        if not self.model:
            raise ValueError("Model not trained yet. Train a model or load one first.")

        X_test_vec = self.vectorizer.transform(X_test)
        if self.model_type == "dnn":
            return self.model.predict(X_test_vec.toarray())
        else:
            return self.model.predict(X_test_vec)

    def save_model(self, file_path):

        ## write an empty file to location first if it doesn't exist

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            ## write an empty file

            with open(file_path, "wb") as f:
                pickle.dump("", f)

        if self.model_type == "dnn":

            self.model.save(file_path)
        else:
            joblib.dump((self.model, self.vectorizer), file_path)

    def load_model(self, file_path):
        if self.model_type == "dnn":
            self.model = tf.keras.models.load_model(file_path)
        else:
            self.model, self.vectorizer = joblib.load(file_path)
