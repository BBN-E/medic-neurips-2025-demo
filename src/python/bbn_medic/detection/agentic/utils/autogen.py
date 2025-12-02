import re

from bbn_medic.detection.agentic.utils.logger import logger, llm_interaction_logger


def postprocessing_and_messages(recipient, messages, sender, config):
    """
    Utility function to print out new messages as they arrive,
    so you can see the entire multi-agent conversation in real-time.
    """
    if "content" in messages[-1]:
        messages[-1]['content'] = interm_llm_resp_postprocessing(messages[-1]['content'])
    # for message in messages:
    #     llm_interaction_logger.info(
    #         f"{message}")
    # llm_interaction_logger.info(f"{'-' * 50}")
    # llm_interaction_logger.info(f"{sender.name} -> {recipient.name}\n\n{messages[-1].get('content', 'NO CONTENT!')}\n{'-' * 50}")
    logger.info(f"{sender.name} -> {recipient.name}\n\n{messages[-1].get('content', 'NO CONTENT!')}\n{'-' * 50}")
    return False, None  # Always continue the flow, do not consume the message


def final_llm_resp_postprocessing(llm_resp_raw_text):
    redundant_markdown_remover = re.compile(r'^```(?:\w+)?\n|```$', flags=re.MULTILINE)
    if isinstance(llm_resp_raw_text, str):
        llm_resp_raw_text = interm_llm_resp_postprocessing(llm_resp_raw_text)
        llm_resp_raw_text = redundant_markdown_remover.sub('', llm_resp_raw_text)
    return llm_resp_raw_text


def interm_llm_resp_postprocessing(llm_resp_raw_text):
    if isinstance(llm_resp_raw_text, str):
        left_think = llm_resp_raw_text.find("</think>")
        if left_think >= 0:
            llm_resp_raw_text = llm_resp_raw_text[left_think + len("</think>"):]
    return llm_resp_raw_text


class TerminateMessageCapture:
    def __init__(self, worker_agent_name, terminate_agent_name, terminate_string):
        self.worker_agent_name = worker_agent_name
        self.terminate_agent_name = terminate_agent_name
        self.terminate_string = terminate_string
        self.captured_output = ""

    def get_captured_output(self):
        return self.captured_output

    def terminate_func_builder(self):
        def terminate_message(msg):
            llm_interaction_logger.info(f"\n{msg.get('content', 'NO CONTENT!')}\n{'-' * 50}")
            if msg[
                "name"] == self.worker_agent_name:  # fallback to last rewrite so Final Text can be extracted
                self.captured_output = msg["content"].replace(self.worker_agent_name, "")
            if msg["name"] == self.terminate_agent_name and self.terminate_string in msg["content"].upper() \
                    and len(self.captured_output) > 0:
                return True
            return False

        return terminate_message
