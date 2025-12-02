import logging
import sys

from tqdm.auto import tqdm

from bbn_medic.common.Answer import Answer
from bbn_medic.io.io_utils import JSONLGenerator, fopen
from bbn_medic.segmentation.Segmenter import Segmenter
from bbn_medic.utils.text_utils import post_process_text

logger = logging.getLogger("bbn_medic.generation.AnswerGenerator")


class AnswerGenerator:
    def generate(input_file,
                 output_file,
                 model_instance,
                 model_temp,
                 num_answers=1,
                 segment_into_sentences=False):

        segmenter = Segmenter() if segment_into_sentences else None
        total_retries_quota = 10
        with fopen(output_file, "w") as g:
            for obj in tqdm(JSONLGenerator.read_all(input_file), file=sys.stdout, desc="Generating answers"):
                id = obj.id
                source_text = obj.text
                obj_type = obj.type
                generated_answer_objects = []
                if obj_type != "Prompt":
                    raise ValueError(f"Unsupported type {obj_type}")

                llm_prompt_text = model_instance.get_formatted_input(source_text)
                generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                        num_return_sequences=num_answers,
                                                        max_length=1000
                                                        )
                generated_answers = [
                    post_process_text(x, characters_to_replace_with_space={'"', '\n'}, remove_leading_spaces=True,
                                      remove_trailing_spaces=True) for x in generated_text]
                while total_retries_quota > 0 and sum(len(i) for i in generated_answers) < 1:
                    print(f"We got empty answer {generated_answers}. Retrying. Quota left {total_retries_quota}")
                    total_retries_quota -= 1
                    generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                            num_return_sequences=num_answers)
                    generated_answers = [
                        post_process_text(x, characters_to_replace_with_space={'"', '\n'}, remove_leading_spaces=True,
                                          remove_trailing_spaces=True) for x in generated_text]
                for x in generated_answers:
                    if len(x) > 0:
                        generated_answer_objects.append(Answer(text=x, metadata={f"{obj_type}_metadata": obj.metadata,"prompt": llm_prompt_text,
                                                                                 f"{obj_type}_source_id": id},
                                                               prompt_id=id,
                                                               chatbot_name=model_instance.get_model_name(),
                                                               segment_into_sentences=segment_into_sentences,
                                                               segmenter_object=segmenter))
                    else:
                        logger.warning(f"Empty answer got for {source_text}")
                for generated_answer in generated_answer_objects:
                    g.write(generated_answer.to_json() + "\n")
                    g.flush()
