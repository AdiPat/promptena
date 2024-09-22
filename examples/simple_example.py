import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from promptena.prompt_classifier import PromptContextClassifier

print("PROMPTENA0: Simple Example (nothing fancy)")


def run_model_example(model):
    print("Creating classifier...")

    classifier = PromptContextClassifier(model_type=model)

    print("Loading model...")

    file_path = f"./models/{model}.pkl"

    if model == "dnn":
        file_path = f"./models/{model}.keras"

    classifier.load_model(file_path=file_path)

    print("Model loaded.")

    prompt = "What is the best feature of the Samsung Galaxy S21?"

    print("Prompt: ", prompt)

    prediction = classifier.predict([prompt])

    response = "Yes" if prediction[0] else "No"

    print("Prediction: Has Enough Context?", response)


run_model_example("svm")
