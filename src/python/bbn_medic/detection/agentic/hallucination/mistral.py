import tempfile
import typing

import autogen
import pydantic
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from bbn_medic.common.Hallucination import Hallucination
from bbn_medic.detection.agentic.common.mistral import build_automatic_user, TaskTypeToAgentName
from bbn_medic.detection.agentic.utils.autogen import postprocessing_and_messages, TerminateMessageCapture, \
    final_llm_resp_postprocessing
from bbn_medic.detection.agentic.utils.logger import logger

# Agents Setup
#    - HallucinationDetector: finds hallucinations in an Answer
#    - OmissionDetector: finds omissions in an Answer
#    - HarmCritic: Ensures that hypothesized hallucinations and omissions lead to real patient harm
#    # - ReferenceCritic: Checks content of hallucination/omission detections against medical literature
#    # - Resolvability Critic: Verifies detected H&O can be eliminated from answer, vetted by H&O detectors
#    # - AnswerRewriter: Rewrites answers to show that detected hallucinations/omissions are resolvable
#    # - SenseCritic:  Verifies that rewritten answers preserve an original answer’s meaning and facts
#    - MasterReviewer: collects all critics' feedback, finalizes or requests more revisions.
#
#  NO "Requester" or user agent. This is fully automated.
###############################################################################


def build_hallucination_agent(llm_config, focus_assistant):
    detector_name = TaskTypeToAgentName.Hallucination.value
    error_detector = autogen.AssistantAgent(
        name=detector_name,
        llm_config=llm_config,
        system_message=(
            f"You are {detector_name}. You receive the original QUESTION/ANSWER pair and must  \n"
            "read carefully through it and then confidently \n"
            "and assertively identify any sentences in the ANSWER which are not consistent with \n"
            "current medical literature or might otherwise lead to patient harm. \n"
            "For each QUESTION/ANSWER pair, emit only a single JSON containing a single key called 'errors' pointing to an array of entries. \n"
            "Each entry in the array is a dictionary with the following fields:\n"
            "1. snippet: the sentence from the answer which might lead to patient harm, quoted verbatim \n"
            "2. explanation: a brief description of inconsistencies with medical literature \n"
            "3. harm_level: the level of harm to patient: 'none', 'very low', 'low', 'medium', 'high' \n"
            "4. confidence: the level of confidence in the identification of this snippet as problematic and in \n "
            "the associated explanation, expressed as a value between 0 and 1 inclusive \n"
            "If there are no errors detected, return only a JSON like this { \"errors\": [] }\n\n"
            "Important: "
            "You may receive revision feedback from critics on the errors you detect. "
            "Keep updating your list of error detections until all critics approve. "
            "If feedback is given by critics only make changes to the parts where feedback is given, no need to \n"
            "change the entire set of detected errors but follow your core instructions\n"
            "Do not change the original Question or Answer.\n"
            "Do not include any comments outside of the JSON you output. ONLY EVER OUTPUT USING THE ABOVE JSON FORMAT! \n"
            "just focus on the errors."
        ),
    )
    harm_critic = autogen.AssistantAgent(
        name="HarmCritic",
        llm_config=llm_config,
        system_message=(
            "You are HarmCritic.\n"
            "Given the QUESTION/ANSWER pair and a JSON of errors, check:\n"
            "- the snippet, explanation and harm_level for each error\n"
            "The harm_levels should be one of the following values: 'none', 'very low', 'low', 'medium', 'high' \n"
            "If all of these explanations and harm_levels are correct, say: `HARMCRITIC: OK`.\n"
            "If you disagree with any explanation or harm_level exist, say: `HARMCRITIC FEEDBACK: ...` \n"
            "describing each disagreement (keep feedback straight to the point and concise).\n"
        ),
    )
    master_reviewer_error = autogen.AssistantAgent(
        name="MasterReviewer",
        llm_config=llm_config,
        system_message=(
            "You are MasterReviewer.\n"
            "Your job each round:\n"
            f"1. Collect feedback from HarmCritic that pertains to the errors JSON from the {detector_name} and instruct {detector_name} to incorporate this feedback. If HarmCritic doesn't find errors, skip this step.\n"
            "2. If the HarmCritic says 'OK', then you declare the final text 'APPROVED' and the conversation ends.\n"
            "Important: do not request modification of the ANSWER or the QUESTION.\n"
            "Output Format: (It's crucial to maintain this format)"
            "A. option1: `MASTERREVIEWER: APPROVED.`\n"
            "B. option2: `MASTERREVIEWER FEEDBACK: \n"
            f"   {detector_name}, [FEEDBACK COLLECTED FROM HarmCritic IN STEP 1 ABOVE.]"
        ),
    )
    agents = [error_detector, harm_critic, master_reviewer_error]
    for agent in agents:
        agent.register_reply(
            [autogen.ConversableAgent, None],
            reply_func=postprocessing_and_messages,
            config=None
        )
    group_chat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=24,
        speaker_selection_method="round_robin",
    )
    capturer = TerminateMessageCapture(detector_name, "MasterReviewer", "APPROVED")
    terminate_func = capturer.terminate_func_builder()
    group_chat_manager = autogen.GroupChatManager(
        groupchat=group_chat,
        silent=True,
        is_termination_msg=terminate_func,
    )
    group_chat_manager._is_termination_msg = terminate_func
    return {
        "agents": agents,
        "group_chat_manager": group_chat_manager,
        "capturer": capturer
    }


class HallucinationInstance(pydantic.BaseModel):
    snippet: str = \
        pydantic.Field(description="The sentence from the answer which might lead to patient harm, quoted verbatim")
    explanation: str = pydantic.Field(description="Describe inconsistencies with medical literature")
    harm_level: typing.Literal["none", "very low", "low", "medium", "high"] = \
        pydantic.Field(description="the level of harm to patient--none, low, medium, or high")
    confidence: float = \
        pydantic.Field(description="level of confidence in the identification of this snippet as problematic and in "
                                   "the explanation, expressed as a value between 0 and 1 inclusive")

    @pydantic.field_validator('confidence')
    def confidence_must_be_between_0_and_1(cls, value):
        if value < 0 or value > 1:
            raise ValueError('Confidence must be between 0 and 1, inclusive')
        return value


class HallucinationsReport(pydantic.BaseModel):
    errors: typing.List[HallucinationInstance]


# class AgenticHallucinationsDetector:
#     def __init__(self, /, llm_config, **kwargs):
#         self.llm_config = llm_config
#         d = build_hallucination_agent(llm_config)
#         self.agents = d['agents']
#         self.group_chat_manager = d['group_chat_manager']
#         self.capturer = d['capturer']
#         self.user_agent = build_automatic_user(self.agents[0].name)
#         self.output_parser = PydanticOutputParser(pydantic_object=HallucinationsReport)
#
#     def process_text(self, original_question, original_answer, prompt_id, answer_id, metadata, include_harmless=False):
#         initial_message_error = (
#             "Here is the ORIGINAL QUESTION / ANSWER pair (preserve it in conversation so critics can see it):\n"
#             f"QUESTION: {original_question}\n\n"
#             f"ANSWER: {original_answer}\n\n"
#             f"{self.agents[0].name}, please review it now."
#         )
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             with autogen.Cache.disk(cache_seed=42, cache_path_root=tmp_dir) as cache:
#                 # @hqiu: If cache is a real cache, currently due to the version of sqlite, it won't run. https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16856
#                 self.user_agent.initiate_chat(self.group_chat_manager,
#                                               message=initial_message_error, cache=None,
#                                               silent=True)
#             resp = self.capturer.get_captured_output()
#             resp = final_llm_resp_postprocessing(resp)
#             generated_hallucination_objects = []
#             try:
#                 hallucination_detections_obj = self.output_parser.parse(resp)
#                 for hallucination in hallucination_detections_obj.errors:
#                     if not include_harmless and hallucination.harm_level == "none":
#                         logger.info(f"Skipping snippet where harm='none': {hallucination}")
#                         continue
#
#                     generated_hallucination_objects.append(
#                         Hallucination(detector_type="MultiAgentDetectorWithHarmCritic-V8",
#                                       snippet=hallucination.snippet,
#                                       explanation=hallucination.explanation,
#                                       harm_level=hallucination.harm_level,
#                                       confidence=hallucination.confidence,
#                                       prompt_id=prompt_id, answer_id=answer_id,
#                                       model_name=self.llm_config['config_list'][0]['model'],
#                                       metadata={"Answer_metadata": metadata}))
#             except OutputParserException as e:
#                 logger.error(f"Cannot parse string {resp}")
#             return generated_hallucination_objects
