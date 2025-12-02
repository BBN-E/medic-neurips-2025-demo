import argparse
import logging
import traceback

from bbn_medic.detection.agentic.multi_agent_detection_100_truncate import client_main, generate_unique_port
from bbn_medic.detection.agentic.utils.llm_service import start_server_with_readiness_check, kill_server
from bbn_medic.detection.agentic.utils.logger import logger

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s {P%(process)d:%(module)s:%(lineno)d} %(levelname)-8s %(message)s',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--focus_task", type=str, required=True)
    parser.add_argument("--agent_group_mode", type=str, required=True)
    parser.add_argument("--output_file", type=str, help="Path to save the generated file", required=True)
    parser.add_argument("--model_path", type=str, help="Path to model", required=True)
    parser.add_argument("--temperature", type=float, default=1.0, required=True)
    parser.add_argument("--readiness_check_timeout", type=int, default=2400)

    args = parser.parse_args()
    logger.info(f"args: {args}")
    port = generate_unique_port(args.output_file)
    server_pid = start_server_with_readiness_check(args.model_path, port, timeout=args.readiness_check_timeout)
    if server_pid is None:
        raise RuntimeError("Cannot bring up backend LLM service")
    try:
        client_main(args.prompts_file, args.answers_file, args.focus_task, args.agent_group_mode, args.model_path,
                    f"http://localhost:{port}/v1", args.temperature, args.output_file)

    except Exception as e:
        logger.exception(traceback.format_exc())
        kill_server(server_pid)
        raise e
    kill_server(server_pid)
