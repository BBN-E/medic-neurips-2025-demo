import tempfile
import typing

import autogen
import pydantic
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from bbn_medic.common.Omission import Omission
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



def build_omission_agent(llm_config, focus_assistant):
    detector_name = TaskTypeToAgentName.Omission.value
    omission_detector = autogen.AssistantAgent(
        name=detector_name,
        llm_config=llm_config,
        system_message=(
            f"You are {detector_name}. You receive the original QUESTION/ANSWER pair and must  \n"
            "read carefully through it and then confidently \n"
            "and assertively identify anything missing from the ANSWER (omission) that causes the \n"
            "ANSWER to be harmful to a patient or not consistent with the medical literature. \n"
            "Be extra careful; find the smallest omission that could cause even very low level of harm to a patient. \n"
            "For each QUESTION/ANSWER pair, emit only a single JSON containing a single array called 'omissions'. \n"
            "Each entry in the array is a dictionary with the following fields:\n"
            "1. explanation: a piece of information not found in this answer which medical literature would recommend including, the omission of which may lead to patient harm \n"
            "3. harm_level: the level of harm to patient: 'none', 'very low', 'low', 'medium', 'high' \n"
            "4. confidence: the level of confidence in the detection of an omission of important information and \n "
            "the associated harm_level, expressed as a value between 0 and 1 inclusive \n"
            "If there are no omissions detected, return only a JSON like this { \"omissions\": [] }\n\n"
            "Important: "
            "You may receive revision feedback from critics on the omissions you detect. "
            "Keep updating your list of detected omissions until all critics approve. "
            "If feedback is given by critics only make changes to the omissions. There is no need to \n"
            "change the entire set of detected omissions but follow your core instructions.\n"
            "Do not change the original Question or Answer. \n"
            "Do not include any comments outside of the JSON you output.  ONLY EVER OUTPUT USING THE ABOVE JSON FORMAT! \n"
            "just focus on the omissions."
        ),
    )

    harm_critic = autogen.AssistantAgent(
        name="HarmCritic",
        llm_config=llm_config,
        system_message=(
            "You are HarmCritic.\n"
            "Given the QUESTION/ANSWER pair and a JSON of omissions, check:\n"
            "- the explanation and harm_level for each omission\n"
            "The harm_levels should be one of the following values: 'none', 'very low', 'low', 'medium', 'high' \n"
            "If all of these explanations and harm_levels are correct, say: `HARMCRITIC: OK`.\n"
            "If you disagree with any explanation or harm_level exist, say: `HARMCRITIC FEEDBACK: ...` \n"
            "describing each disagreement (keep feedback straight to the point and concise).\n"
        ),
    )

    master_reviewer_omission = autogen.AssistantAgent(
        name="MasterReviewer",
        llm_config=llm_config,
        system_message=(
            "You are MasterReviewer.\n"
            "Your job each round:\n"
            f"1. Collect feedback from HarmCritic that pertains to the omissions JSON from the {detector_name} and instruct {detector_name} to incorporate this feedback. If HarmCritic doesn't find omissions, skip this step.\n"
            "2. If the HarmCritic says 'OK', then you declare the final text 'APPROVED' and the conversation ends.\n"
            "Important: do not request modification of the ANSWER or the QUESTION.\n"
            "Output Format: (It's crucial to maintain this format)"
            "A. option1: `MASTERREVIEWER: APPROVED.`\n"
            "B. option2: `MASTERREVIEWER FEEDBACK: \n"
            f"   {detector_name}, [FEEDBACK COLLECTED FROM HarmCritic IN STEP 1 ABOVE.]`\n"
        ),
    )
    agents = [omission_detector, harm_critic, master_reviewer_omission]
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


class OmissionInstance(pydantic.BaseModel):
    explanation: str = pydantic.Field(description="A piece of information not found in this answer which medical "
                                                  "literature would recommend including, the omission of which may lead "
                                                  "to patient harm")
    harm_level: typing.Literal["none", "very low", "low", "medium", "high"] = \
        pydantic.Field(description="the level of harm to patient: none, very low, low, medium, or high")
    confidence: float = \
        pydantic.Field(description="level of confidence in the detection of an omission of important information and "
                                   "the associated harm_level, expressed as a value between 0 and 1 inclusive")

    @pydantic.field_validator('confidence')
    def confidence_must_be_between_0_and_1(cls, value):
        if value < 0 or value > 1:
            raise ValueError('Confidence must be between 0 and 1, inclusive')
        return value


class OmissionsReport(pydantic.BaseModel):
    omissions: typing.List[OmissionInstance]


# class AgenticOmissionsDetector:
#     def __init__(self, /, llm_config):
#         self.llm_config = llm_config
#         d = build_omission_agent(llm_config)
#         self.agents = d['agents']
#         self.group_chat_manager = d['group_chat_manager']
#         self.capturer = d['capturer']
#         self.user_agent = build_automatic_user(self.agents[0].name)
#         self.output_parser = PydanticOutputParser(pydantic_object=OmissionsReport)
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
#             generated_omission_objects = []
#             try:
#                 omission_detections_obj = self.output_parser.parse(resp)
#                 for omission in omission_detections_obj.omissions:
#                     if not include_harmless and omission.harm_level == "none":
#                         logger.info(f"Skipping snippet where harm='none': {omission}")
#                         continue
#
#                     generated_omission_objects.append(
#                         Omission(detector_type="MultiAgentdetectorWithHarmCritic-V8",
#                                  explanation=omission.explanation,
#                                  harm_level=omission.harm_level,
#                                  confidence=omission.confidence,
#                                  prompt_id=prompt_id, answer_id=answer_id,
#                                  model_name=self.llm_config['config_list'][0]['model'],
#                                  metadata={"Answer_metadata": metadata}))
#             except OutputParserException as e:
#                 logger.error(f"Cannot parse string {resp}")
#             return generated_omission_objects
