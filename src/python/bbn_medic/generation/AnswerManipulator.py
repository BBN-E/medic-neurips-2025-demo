import sys
import random
import spacy
from tqdm.auto import tqdm

from bbn_medic.common.Answer import Answer
from bbn_medic.common.Prompt import Prompt
from bbn_medic.common.Hallucination import Hallucination
from bbn_medic.common.Omission import Omission
from bbn_medic.io.io_utils import JSONLGenerator, fopen
from bbn_medic.segmentation.Segmenter import Segmenter
from bbn_medic.utils.text_utils import post_process_text


class AnswerManipulator:
    def generate(input_file,
                 output_file,
                 omissions_proportion,
                 hallucinations_proportion,
                 model_instance,
                 model_temp,
                 hallucinations_reference_file=None,
                 omissions_reference_file=None,
                 segment_level=False,
                 replace_statements_with_hallucinations=False):

        population = [False, True]
        hallucination_weights = [1 - hallucinations_proportion, hallucinations_proportion]
        omission_weights = [1 - omissions_proportion, omissions_proportion]

        reference_hallucinations_correspondence = {}
        reference_omissions_correspondence = {}
        random.seed(42)
        segmenter = Segmenter() if segment_level else None

        with fopen(output_file, "w") as g:
            for obj in tqdm(JSONLGenerator.read_all(input_file), file=sys.stdout, desc="Manipulating answers"):
                id = obj.id
                source_text = obj.text
                obj_type = obj.type
                prompt_id = obj.prompt_id
                if obj_type != "Answer":
                    raise ValueError(f"Unsupported type {obj_type}")

                if segment_level and "segments" not in obj.metadata:
                    raise ValueError(f"Answer with id {id} should have been segmented")

                # We introduce omissions first
                introduce_answer_level_omission = random.choices(population, weights=omission_weights)[0]
                if introduce_answer_level_omission:
                    omission_prompt_text = model_instance.get_formatted_input(f"Generate a short summary for the following sentences, without details:\n\"{source_text}\"")
                    generated_text = model_instance.forward(omission_prompt_text, temperature=model_temp, num_return_sequences=1)
                    generated_answers = [post_process_text(x,
                                                           apply_unidecode=True,
                                                           characters_to_replace_with_space={'"', '\n'},
                                                           remove_leading_spaces=True,
                                                           remove_trailing_spaces=True) for x in generated_text]
                    change_types = AnswerManipulator.update_change_types(obj.metadata, ["answer_level_omission"])
                    new_obj = Answer(text=generated_answers[0], prompt_id=prompt_id, chatbot_name=model_instance.get_model_name(),
                                     segment_into_sentences=segment_level, segmenter_object=segmenter,
                                     metadata={f"{obj_type}_metadata": obj.metadata,
                                               f"{obj_type}_source_id": id,
                                               f"{obj_type}_source_text": source_text,
                                               "change_types": change_types})
                    answer_id = new_obj.id
                    reference_omissions_correspondence[(prompt_id, id)] = None  # The value of this dict is updated later
                else:
                    new_obj = obj

                # We introduce hallucinations next
                if not segment_level:
                    introduce_answer_level_hallucination = random.choices(population, weights=hallucination_weights)[0]
                    if introduce_answer_level_hallucination:
                        hallucination_prompt_text = model_instance.get_formatted_input(f"Generate a negated version of the following sentences:\n\"{new_obj.text}\"")
                        generated_text = model_instance.forward(hallucination_prompt_text, temperature=model_temp, num_return_sequences=1)
                        generated_answers = [post_process_text(x,
                                                               apply_unidecode=True,
                                                               characters_to_replace_with_space={'"', '\n'},
                                                               remove_leading_spaces=True,
                                                               remove_trailing_spaces=True) for x in generated_text]
                        change_types = AnswerManipulator.update_change_types(new_obj.metadata, ["answer_level_hallucination"])
                        new_obj = Answer(text=generated_answers[0], prompt_id=prompt_id, chatbot_name=model_instance.get_model_name(),
                                         segment_into_sentences=segment_level, segmenter_object=segmenter,
                                         metadata={f"{obj_type}_metadata": obj.metadata,
                                                   f"{obj_type}_source_id": id,
                                                   f"{obj_type}_source_text": source_text,
                                                   "change_types": change_types})
                        answer_id = new_obj.id
                        reference_hallucinations_correspondence[(prompt_id, id, None)] = (new_obj, obj.text)
                else:
                    segments = new_obj.metadata["segments"]
                    hallucinated_segments = set()
                    segment_index = 0
                    change_types = AnswerManipulator.update_change_types(new_obj.metadata, [])
                    new_obj = Answer(text="", prompt_id=prompt_id, chatbot_name=model_instance.get_model_name(),
                                     metadata={f"{obj_type}_metadata": obj.metadata,
                                               f"{obj_type}_source_id": id,
                                               f"{obj_type}_source_text": source_text,
                                               "change_types": change_types,
                                               "segments": []})
                    for segment in segments:
                        introduce_hallucination = random.choices(population, weights=hallucination_weights)[0]
                        if introduce_hallucination:
                            hallucination_prompt_text = model_instance.get_formatted_input(f"Generate one sentence, with commas instead of periods, which has the opposite meaning of the following sentence:\n\"{segment['text']}\"")
                            generated_text = model_instance.forward(hallucination_prompt_text, temperature=model_temp, num_return_sequences=1)
                            generated_answers = [post_process_text(x,
                                                                   apply_unidecode=True,
                                                                   characters_to_replace_with_space={'"', '\n'},
                                                                   remove_leading_spaces=True,
                                                                   remove_trailing_spaces=True) for x in generated_text]
                            new_obj.update_with_segment(generated_answers[0] + " ")
                            new_obj.metadata["change_types"].append("segment_level_hallucination")
                            hallucinated_segments.add(segment_index)
                            segment_index += 1
                        if not introduce_hallucination or not replace_statements_with_hallucinations:
                            new_obj.update_with_segment(segment["text"] + " ")
                            new_obj.metadata["change_types"].append("segment_level_no_change")
                            segment_index += 1
                    answer_id = new_obj.id
                    for index_of_segment_with_hallucination in sorted(hallucinated_segments):
                        reference_hallucinations_correspondence[(prompt_id, id, index_of_segment_with_hallucination)] = (new_obj, segment['text'])

                # Updating the omissions correspondence with the *latest* new_obj
                if (prompt_id, id) in reference_omissions_correspondence:
                    reference_omissions_correspondence[(prompt_id, id)] = (new_obj, obj.text)
                g.write(new_obj.to_json() + "\n")

        if reference_hallucinations_correspondence and hallucinations_reference_file is not None:
            with fopen(hallucinations_reference_file, "w") as g:
                for (prompt_id, id, segment_id), (reference_hallucination, original_text) in reference_hallucinations_correspondence.items():
                    answer_id = reference_hallucination.id
                    if segment_id is None:
                        hallucination_snippet = reference_hallucination.text
                    else:
                        hallucination_snippet = reference_hallucination.metadata["segments"][segment_id]["text"]
                    hallucination_obj = Hallucination(detector_type="ArtificialHallucinationReference",
                                                      snippet=hallucination_snippet,
                                                      explanation=hallucination_snippet,
                                                      harm_level="medium",
                                                      confidence=1.0,
                                                      prompt_id=prompt_id,
                                                      answer_id=answer_id,
                                                      segment_id=segment_id,
                                                      model_name=model_instance.get_model_name(),
                                                      metadata={"Answer_text": original_text})
                    g.write(hallucination_obj.to_json() + "\n")
        if reference_omissions_correspondence and omissions_reference_file is not None:
            with fopen(omissions_reference_file, "w") as g:
                for (prompt_id, answer_id), (reference_omission, original_text) in reference_omissions_correspondence.items():
                    answer_id = reference_omission.id
                    omission_text = reference_omission.text
                    omission_obj = Omission(detector_type="ArtificialOmissionReference",
                                            explanation=omission_text,
                                            harm_level="medium",
                                            confidence=1.0,
                                            prompt_id=prompt_id,
                                            answer_id=answer_id,
                                            model_name=model_instance.get_model_name(),
                                            metadata={"Answer_text": original_text})
                    g.write(omission_obj.to_json() + "\n")

    def update_change_types(metadata, change_types):
        prev_change_types = metadata.get("change_types", [])
        return prev_change_types + change_types
