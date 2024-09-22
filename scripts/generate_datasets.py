import sys
import os
from dotenv import load_dotenv
import traceback
import csv
import random

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from promptena.llm import LLM, LLMOptions, GenerateOptions
from pydantic import BaseModel, Field
import json


MAX_ROWS = 10
DATASET_FILE_NAME = "promptena_core_dataset"

llm_options = LLMOptions()
llm = LLM(llm_options)


class PromptenaCoreDatasetRowItem(BaseModel):
    prompt: str
    is_context_sufficient: bool
    reason: str
    confidence: float


def generate_case(is_context_sufficient: bool) -> PromptenaCoreDatasetRowItem:

    context_prompt = "There is sufficient context so set 'is_context_sufficient' to True and provide a prompt with context. This prompt doesn't require fetching of additional data."

    if not is_context_sufficient:
        context_prompt = "There is insufficient context so set 'is_context_sufficient' to False and provide a prompt without context. This prompt requires fetching of additional data."

    num_sentences = random.randint(1, 10)

    prompt = f"""
                I'm building a RAG system. 
                For that I need to train a Deep Neural Network to classify whether a prompt has enough 'context' or not. Context is required when the response to the prompt requires information that the LLM does not know or understand. 
                The prompt is between 1 to 10 sentences (THIS IS VERY IMPORTANT: VARY THE PROMPT SIZE BETWEEN 1 TO 10 SENTENCES). I have given you the schema that I want your answer to be in. Give me 20 prompts and their 'is_context_sufficient' status. 
                Vary the topics between different business domains and problem statements. Be as creative as you can. You should mimic how an actual user would prompt an LLM.

                {context_prompt}
             """

    system_prompt = "You are a Synthetic Data Generation AI Agent. Generate high-quality and relevant synthetic data for the given task."

    generate_options = GenerateOptions(
        prompt=prompt,
        system_prompt=system_prompt,
        response_schema=PromptenaCoreDatasetRowItem,
        verbose=True,
    )

    return llm.generate_object(options=generate_options)


def generate_dataset():
    positive_cases = MAX_ROWS // 2
    negative_cases = MAX_ROWS - positive_cases

    dataset = []

    for i in range(positive_cases):
        obj = generate_case(is_context_sufficient=True)
        dataset.append(obj.model_dump())

    for i in range(negative_cases):
        obj = generate_case(is_context_sufficient=False)
        dataset.append(obj.model_dump())

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

            # incase write with json was called before, don't add duplicate entries
            new_entries = []

            with open(f"./datasets/{DATASET_FILE_NAME}.json", "r") as f:
                cur_all_entries = json.load(f)
                for entry in dataset:
                    if entry["prompt"] not in [e["prompt"] for e in cur_all_entries]:
                        new_entries.append(entry)

            write_dataset_to_file(new_entries, format="json")

            all_data = []
            with open(f"./datasets/{DATASET_FILE_NAME}.json", "r") as f:
                all_data = json.load(f)
            with open(f"./datasets/{DATASET_FILE_NAME}.csv", "w") as f:
                dict_writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(all_data)
                print("write_dataset: wrote", len(all_data), " rows to file. ")

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
        # write_dataset_to_file(dataset=dataset, format="json")
        write_dataset_to_file(dataset=dataset, format="csv")
    except Exception as e:
        print("generate_datasets: script failed. ")
        traceback.print_exc()


if __name__ == "__main__":
    run()
