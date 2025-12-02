import logging
import sys
import typing

from bbn_medic.io.io_utils import JSONLGenerator, fopen
from bbn_medic.llms.huggingface_pipeline import MyHuggingFaceModel
from bbn_medic.utils.text_utils import post_process_text
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from tqdm.auto import tqdm
import json

global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaselineTreatmentClassifier.py")


class BaselineTreatmentClassifier:

    @staticmethod
    def get_task_description_for_system_prompt():
        return """Annotate the clinician response with answers to the following questions. Answer each question with [YES] or [NO].

        MANAGE
        Does the writer provides recommendations for the patient to self-manage at home? Statements that the writer will prescribe medication is ACT, not MANAGE.  

        VISIT 
        Does the writer recommend that the patient comes into clinic, urgent care, or ED? 

        RESOURCE
        Does the response suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? 
        Suggestions for non-clinical resources that do not require a referral or prescription do not count and the answer should be no.

        Your response should be structured like: 
        MANAGE [YES/NO] 
        VISIT [YES/NO] 
        RESOURCE [YES/NO] 

        Here is an example:

        Example Of Clinician Response:

        The provided information is sufficient to address the patient's concerns.

        Dear [Patient's Name],

        I understand that you have been experiencing persistent abdominal bloating and discomfort for the past week. It is possible that these symptoms could be related to your ongoing chemotherapy treatment with paclitaxel and carboplatin, as these medications can sometimes cause gastrointestinal side effects.

        However, given your history of stage III ovarian cancer and recent debulking surgery, it is important to monitor these symptoms closely and rule out other potential causes. I recommend that you schedule an appointment with our office so we can further evaluate your symptoms and determine if any additional tests or interventions are needed.

        In the meantime, you can try some over-the-counter gas relief medications and maintain a balanced diet with adequate hydration to help alleviate your discomfort. Please do not hesitate to reach out if your symptoms worsen or if you have any other concerns.

        Best regards,

        [Your Name]
        Oncologist

        Correct Answer:

        MANAGE [YES]
        VISIT [YES]
        RESOURCE [NO]

        Here is the response you are to evaluate. Please think carefully step by step."""

    def detect(prompts_input_file,
               answers_input_file,
               output_file,
               model_instance,
               model_temp,
               retry_max_attempts=1,
               include_harmless=False):

        class TreatmentAnnotation(BaseModel):
            MANAGE: typing.Literal["YES", "NO"] = Field(description="Does the writer provide recommendations for the patient to self-manage at home?")
            VISIT: typing.Literal["YES", "NO"] = Field(description="Does the writer recommend that the patient comes into clinic, urgent care, or ED?")
            RESOURCE: typing.Literal["YES", "NO"] = Field(description="Does the response suggest a resource such as a lab, test, imaging, or specialist referral?")


        # load prompts for later lookup
        prompt_ids_to_prompts = {}
        for obj in JSONLGenerator.read(prompts_input_file):
            prompt_id = obj.id
            prompt_text = obj.text
            obj_type = obj.type
            if obj_type != "Prompt":
                raise ValueError(f"Unsupported type {obj_type} in prompts_input_file {prompts_input_file}")
            prompt_ids_to_prompts[prompt_id] = obj

        # load langchain-compatible model adapter
        llm = MyHuggingFaceModel(huggingface_tokenizer=model_instance._tokenizer,
                                 huggingface_model=model_instance._model)

        # create prompt templates for expansion
        system_prompt_template = PromptTemplate(
            template=f"""You are a medical expert. Please read carefully through the following clinical advice 
             and {BaselineTreatmentClassifier.get_task_description_for_system_prompt()}""",
            input_variables=[],
            partial_variables={}
        )
        user_prompt_template = PromptTemplate(
            template="""\nClinical answer: {answer}""",
            input_variables=["question", "answer"],
            partial_variables={}
        )

        with fopen(output_file, "w") as g:
            for obj in tqdm(JSONLGenerator.read_all(answers_input_file), file=sys.stdout,
                            desc="Detecting \"manage\" response"):
                # parse the Answer object from jsonl
                answer_id = obj.id
                answer_text = obj.text
                obj_type = obj.type
                generated_hallucination_objects = []
                if obj_type != "Answer":
                    raise ValueError(f"Unsupported type {obj_type} in answers_input_file {answers_input_file}")

                # get the source prompt text (should have been loaded above)
                prompt_id = obj.prompt_id
                if prompt_id not in prompt_ids_to_prompts:
                    raise ValueError(
                        f"Unable to find prompt {prompt_id} referenced in answer {answer_id} in {prompts_input_file}")

                # check if our model supports structured output
                structured_output_supported = False
                try:
                    llm = llm.with_structured_output(TreatmentAnnotation)
                    structured_output_supported = True
                    logger.info("structured output supported by this LLM")
                except NotImplementedError as e:
                    logger.info("structured output not supported by this LLM")
                    output_parser = PydanticOutputParser(pydantic_object=TreatmentAnnotation)
                
                messages = [system_prompt_template.format(),
                            user_prompt_template.format(question=prompt_ids_to_prompts[prompt_id].text,
                                                        answer=answer_text)]
                if structured_output_supported is False:
                    messages.append(HumanMessage(content=output_parser.get_format_instructions()))

                retry_cnt = 0
                output = None
                while True:
                    try:
                        if (
                                output is not None  # must have thrown OutputParserException before; try to mitigate
                                and not (output.content.startswith('{\n') and output.content.endswith('\n}'))
                        ):
                            output.content = f"{{\n{output.content}\n}}"  # common error is omitting surrounding braces
                        else:
                            # Execute the prompt and parse the output
                            output = llm.invoke(messages)

                        output_text = output.content
                        output_text = post_process_text(output_text, remove_leading_spaces=True,
                                                        remove_trailing_spaces=True)
                        annotation_obj = output_parser.parse(output_text)
                        result = {
                            "answer_id": answer_id,
                            "prompt_id": prompt_id,
                            "model_name": model_instance.get_model_name(),
                            "MANAGE": annotation_obj.MANAGE,
                            "VISIT": annotation_obj.VISIT,
                            "RESOURCE": annotation_obj.RESOURCE,
                            "metadata": obj.metadata
                        }
                        g.write(json.dumps(result) + "\n") 
                        
                        output = None
                        break

                    except OutputParserException as e:
                        retry_cnt += 1
                        logger.info("OutputParserException {}".format(str(e)))
                        if retry_cnt > retry_max_attempts:
                            logger.info(
                                f"retry count {retry_cnt} exceeds retry_max_attempts. Returning...\n\n"
                                f"Last output was {output}")
                            break
                        logger.info(
                            "Retry count: {}. Output parsing failed. We got output:\n{}\n".format(retry_cnt, output))
