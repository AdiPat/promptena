import sys
import os
from dotenv import load_dotenv
import traceback
import csv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from promptena.llm import LLM, LLMOptions, GenerateOptions
from pydantic import BaseModel, Field
import json


MAX_ROWS = 10
DATASET_FILE_NAME = "promptena_core_dataset"


class PromptenaCoreDatasetRowItem(BaseModel):
    prompt: str
    is_context_sufficient: bool
    reason: str
    confidence: float


def generate_dataset():
    llm_options = LLMOptions()
    llm = LLM(llm_options)

    prompt = """
                Generate a prompt for a random task. The prompt should be a question that requires a short answer.
                The prompt should be related to a task that a user would like to get help with.
                The purpose of the prompt is that we want to train an ML model to determine if a prompt has enough context or not.
                This is so that we can decide whether more context needs to be fetched from data storage systems or not.  
                The generated prompt length can be between 1 sentence to 10 sentences. 
             """

    system_prompt = "You are a Synthetic Data Generation AI Agent. Generate high-quality and relevant synthetic data for the given task."

    generate_options = GenerateOptions(
        prompt=prompt,
        system_prompt=system_prompt,
        response_schema=PromptenaCoreDatasetRowItem,
        verbose=True,
    )

    dataset = []

    for _ in range(MAX_ROWS):
        response = llm.generate_object(options=generate_options)
        dataset.append(response.model_dump())

    return dataset


def write_dataset_to_file(dataset, format="json") -> None:
    ## not validating data, assumes it's always updated and correct
    if len(dataset) == 0:
        print("No data to write to file.")
        return

    if format == "json":
        os.makedirs("./datasets", exist_ok=True)

        existing_data = []

        # read existing data
        try:
            with open(f"./datasets/{DATASET_FILE_NAME}.json", "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        all_data = existing_data + dataset

        with open(f"./datasets/{DATASET_FILE_NAME}.json", "w") as f:
            json.dump(all_data, f, indent=4)

        print("write_dataset: wrote", len(all_data), " rows to file. ")
    if format == "csv":
        try:
            # check if json file exists
            file_existed_previously = os.path.exists(
                f"./datasets/{DATASET_FILE_NAME}.json"
            )

            write_dataset_to_file(dataset, format="json")
            all_data = []
            with open(f"./datasets/{DATASET_FILE_NAME}.json", "r") as f:
                all_data = json.load(f)
            with open(f"./datasets/{DATASET_FILE_NAME}.csv", "w") as f:
                dict_writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(all_data)

            if not file_existed_previously:
                os.remove(f"./datasets/{DATASET_FILE_NAME}.json")

        except Exception as e:
            print("write_dataset: failed to write to csv. ")
            traceback.print_exc()


def run():
    try:
        print("generate_datasets: generating new dataset. ", {"max_rows": MAX_ROWS})
        dataset = generate_dataset()
        print(
            "generate_datasets: writing dataset to file. ",
            {"rows_generated": len(dataset)},
        )
        write_dataset_to_file(dataset=dataset, format="json")
        write_dataset_to_file(dataset=dataset, format="csv")
    except Exception as e:
        print("generate_datasets: script failed. ")
        traceback.print_exc()


if __name__ == "__main__":
    run()
