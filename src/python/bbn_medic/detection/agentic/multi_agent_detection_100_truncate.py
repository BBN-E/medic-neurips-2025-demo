import enum
import hashlib
import json
import os
import logging

from tqdm.auto import tqdm

from bbn_medic.detection.agentic.hybrid.mistral import AgenticDetector, TaskAgentGroup, TaskTypeToAgentName
from bbn_medic.detection.agentic.utils.llm_service import build_llm_config_block
from bbn_medic.io.io_utils import fopen, JSONLGenerator


def generate_unique_port(unique_output_filename):
    # Hash the path using MD5
    hash_hex = hashlib.md5(unique_output_filename.encode('utf-8')).hexdigest()

    # Convert a portion of the hash to an integer and map it to a valid port range
    port = 32768 + (int(hash_hex[:8], 16) % (60999 - 32768))

    return port


def client_main(input_questions_jsonl, input_answers_jsonl, focus_task, agent_group_mode, model_path, model_api_url,
                temperature,
                output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    llm_config = build_llm_config_block(model_path, model_api_url, temperature)
    detector = AgenticDetector(agent_group_mode=agent_group_mode, focus_task=focus_task, llm_config=llm_config)
    prompt_ids_to_prompts = {}
    for obj in JSONLGenerator.read(input_questions_jsonl):
        prompt_id = obj.id
        prompt_text = obj.text
        obj_type = obj.type
        if obj_type != "Prompt":
            raise ValueError(f"Unsupported type {obj_type} in prompts_input_file {input_questions_jsonl}")
        prompt_ids_to_prompts[prompt_id] = obj

    num_answers = 0
    with fopen(input_answers_jsonl) as f:
        for line in f:
            num_answers += 1
    outputs = []
    with fopen(input_answers_jsonl) as f:  # TODO: replace with proper Answer parsing
        for index, line in tqdm(enumerate(f), desc="Processing JSONL file", unit="lines", total=num_answers):
            obj = json.loads(line)
            answer_id = obj["id"]
            prompt_id = obj["prompt_id"]
            original_question = prompt_ids_to_prompts[prompt_id].text
            original_answer = obj["text"]
            # New processing:
            if len(original_answer.split(" ")) > 100:
                original_answer = " ".join(original_answer.split(" ")[:100])
                logging.info(f"Truncated line {index} to length 100: \n{original_answer}\n\n")
            metadata = obj['metadata']
            outputs.extend(detector.process_text(original_question, original_answer, prompt_id, answer_id, metadata))
    with fopen(output_path, 'w') as f:
        for entry in outputs:
            f.write(f"{entry.to_json()}\n")


def debug_main():
    log_file_handler = logging.FileHandler("/nfs/alletra/projects/care/hqiu/tmp/a.log", mode='w', encoding='utf-8')
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("bbn_medic.detection.agentic.conversation").setLevel(logging.INFO)
    logging.getLogger("bbn_medic.detection.agentic.conversation").addHandler(log_file_handler)
    input_questions_jsonl = "/nfs/alletra/projects/care/hqiu/repos/medic/data/jhu-delivery-1/mental_health_prompts.jsonl"
    input_answers_jsonl = "/nfs/alletra/projects/care/hqiu/repos/medic/experiments/answer_evaluation/expts/name/answers/answers_file.0.jsonl"
    focus_task = "Hallucination"
    agent_group_mode = "HallucinationOnly"
    model_path = "/nfs/nimble/projects/ami/models/Mistral-Nemo-Instruct-2407"
    model_api_url = "http://localhost:4117/v1"
    temperature = 0
    output_path = "/nfs/alletra/projects/care/hqiu/repos/medic/experiments/answer_evaluation/expts/name/hallucination_detection/agentic_hallucination_detections.0.jsonl"
    client_main(input_questions_jsonl, input_answers_jsonl, focus_task, agent_group_mode, model_path, model_api_url, temperature,
                output_path)
    log_file_handler.close()


if __name__ == "__main__":
    debug_main()
