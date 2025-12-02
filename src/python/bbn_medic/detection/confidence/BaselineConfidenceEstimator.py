import logging
import typing

from pydantic import BaseModel, Field, field_validator

from bbn_medic.common.Confidence import Confidence
from bbn_medic.io.io_utils import JSONLGenerator, fopen
from bbn_medic.llms.huggingface_pipeline import MyHuggingFaceModel

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage

global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaselineConfidenceEstimator.py")


class BaselineConfidenceEstimator:

    def detect(prompts_input_file, answers_input_file, output_file, model_instance, model_temp, retry_max_attempts=1):

        class ConfidenceInstance(BaseModel):
            explanation: str = Field(description="Explanation for why the provided answer is or is not written using confident language")
            confidence: float = \
                Field(description="estimated level of confidence indicated by the language of the answer, expressed as a value between 0 and 1 inclusive")

            @field_validator('confidence')
            def confidence_must_be_between_0_and_1(cls, value):
                if value < 0 or value > 1:
                    raise ValueError('Confidence must be between 0 and 1, inclusive')
                return value

        class ConfidencesReport(BaseModel):
            detections: typing.List[ConfidenceInstance]

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
        llm = MyHuggingFaceModel(huggingface_tokenizer=model_instance._tokenizer, huggingface_model=model_instance._model)

        # create prompt templates for expansion
        system_prompt_template = PromptTemplate(
            template="You are a medical expert. Read carefully the following QUESTION and ANSWER pair and determine the level of confidence indicated by the language of the ANSWER.",
            input_variables=[],
            partial_variables={}
        )
        user_prompt_template = PromptTemplate(
            template="""QUESTION: {question}\n ANSWER: {answer}""",
            input_variables=["question", "answer"],
            partial_variables={}
        )

        with fopen(output_file, "w") as g:
            for obj in JSONLGenerator.read(answers_input_file):
                # parse the Answer object from jsonl
                answer_id = obj.id
                answer_text = obj.text
                obj_type = obj.type
                generated_confidence_objects = []
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
                    llm = llm.with_structured_output(ConfidencesReport)
                    structured_output_supported = True
                    logger.info("structured output supported by this LLM")
                except NotImplementedError as e:
                    logger.info("structured output not supported by this LLM")
                    output_parser = OutputFixingParser.from_llm(
                        parser=PydanticOutputParser(pydantic_object=ConfidencesReport),
                        llm=llm)

                messages = [system_prompt_template.format(),
                            user_prompt_template.format(question=prompt_ids_to_prompts[prompt_id].text,
                                                        answer=answer_text)]
                if structured_output_supported is False:
                    messages.append(HumanMessage(content=output_parser.get_format_instructions()))

                retry_cnt = 0
                while True:
                    try:
                        # Execute the prompt and parse the output
                        output = llm.invoke(messages)
                        detections_obj = output_parser.parse(output.content)
                        for confidence in detections_obj.detections:
                            generated_confidence_objects.append(
                                Confidence(explanation=confidence.explanation,
                                           confidence=confidence.confidence,
                                           prompt_id=prompt_id, answer_id=answer_id,
                                           model_name=model_instance.get_model_name(),
                                           metadata={"Answer_metadata": obj.metadata}))
                        break
                    except OutputParserException as e:
                        retry_cnt += 1
                        logger.info("OutputParserException {}".format(str(e)))
                        if retry_cnt >= retry_max_attempts:
                            logger.info("retry count {} exceeds retry_max_attempts. Returning...")
                            return None
                        logger.info(
                            "Retry count: {}. Output parsing failed. We got output:\n{}\n".format(retry_cnt, output))

                for generated_confidence in generated_confidence_objects:
                    g.write(generated_confidence.to_json() + "\n")
