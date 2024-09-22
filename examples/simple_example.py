import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from promptena.prompt_classifier import PromptContextClassifier

print("PROMPTENA0: Simple Example (nothing fancy)")


def run_model_example(model):
    print("Creating classifier...")

    classifier = PromptContextClassifier(model_type=model)

    print("Loading model...")

    is_trained = classifier.train_from_stored_data()

    if not is_trained:
        print("Model not trained. Training model from stored data.")
        sys.exit(1)

    print("Model loaded.")

    prompts = [
        "What is the best feature of the Samsung Galaxy S21?",
        "What is the capital of France?",
        "What is the time in New York?",
        "What is the weather in London?",
        "What is the core value of the company?",
    ]

    predictions = classifier.predict(prompts)

    print("Predictions: ", predictions)

    for i in range(len(prompts)):
        prompt = prompts[i]
        prediction = predictions[i]
        print("Prompts: ", prompt)
        print("Prediction: ", prediction)
        response = None

        if model == "dnn":
            response = "Yes" if prediction[0] else "No"
        else:
            response = "Yes" if prediction else "No"

        print("Has Enough Context?", response)


run_model_example("svm")

# run_model_example("dnn")
