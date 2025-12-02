import os
import signal
import subprocess
import sys
import shlex
import threading
import time
import traceback

import torch
from bbn_medic.detection.agentic.utils.logger import logger


def log_subprocess_output(pipe, call_back):
    for line in iter(pipe.readline, b''):  # b'\n'-separated lines
        if len(line.strip()) > 0:
            call_back(line)
    pipe.close()


def start_server(model_path, model_api_port):
    python_bin = sys.executable
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    logger.info(f"model info: {model_path}, {model_dir}, {model_name}")
    # the agentic code uses an external serverized process to load and interact with LLMs.
    # the first version uses the text-generation-webui which has a very specific way to invoke.
    # this is implemented below, e.g. it requires a python subprocess run from a specific directory.
    # there is a plan to refactor this more comprehensively across bbn_medic (ARPAHCRE-58) as well as
    # examples for how to do this using VLLM instead of open-web-ui (ARPAHCRE-67).
    server_process = subprocess.Popen(
        [python_bin, "server.py", "--extensions", "openai", "--listen", "--listen-host", "0.0.0.0",
         "--listen-port",
         str(model_api_port + 1),
         "--api", "--api-port", str(model_api_port), "--model", model_name, "--model-dir", model_dir],
        cwd="/nfs/nimble/projects/hiatus/hqiu/frozen/text-generation-webui/769eee1",
        # TODO coppy from HIATUS to MEDIC project space
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    chat_template_extra = ""
    # if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
    #     with open(os.path.join(model_path, "tokenizer_config.json")) as fp:
    #         d = json.load(fp)
    #         if "chat_template" in d:
    #             tmp_file = tempfile.NamedTemporaryFile(delete=False)
    #             tmp_file.write(d['chat_template'].encode("utf-8"))
    #             tmp_file.close()
    #             chat_template_extra = f"--chat-template {tmp_file.name}"

    # server_process = subprocess.Popen(
    #     shlex.split(
    #         f"/nfs/alletra/projects/care/apps/miniforge3/envs/medic-060525-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model {model_path} --tokenizer {model_path} --dtype auto --max_model_len 40000 --port {str(model_api_port)} --host :: --tensor_parallel_size {torch.cuda.device_count()} {chat_template_extra}"),
    #     TODO, install vllm using official instruction to medic conda env. We may need to get rid of conda-forge
        # preexec_fn=os.setsid,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        # text=True,
    # )
    logger.info(f"Server started with PID: {server_process.pid}")

    t = threading.Thread(target=log_subprocess_output, args=(server_process.stdout, logger.info))
    t.daemon = True  # thread dies with the program
    t.start()
    t = threading.Thread(target=log_subprocess_output, args=(server_process.stderr, logger.info))
    t.daemon = True  # thread dies with the program
    t.start()
    logger.info(f"Logger registered for PID: {server_process.pid}")
    return server_process


def kill_server(server_process):
    logger.info("Killing the server...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        logger.info("Server killed.")
    except Exception as e:
        logger.exception(traceback.format_exc())


def signal_kill_server(server_process):
    kill_server(server_process)
    exit(-1)


def start_server_with_readiness_check(model_path, model_api_port, timeout=240, check_interval=10):
    num_quota_retries = timeout / check_interval
    server_process = start_server(
        model_path=model_path,
        model_api_port=model_api_port,
    )
    signal.signal(signal.SIGINT, lambda signum, frame: signal_kill_server(server_process))
    while num_quota_retries > 0:
        if server_process.returncode is not None:
            # While waiting for the server to up, it dies.
            return None
        try:
            response = subprocess.check_output(["curl", "-s", f"http://localhost:{model_api_port}/v1/"])
            if response.decode("utf-8") == '{"detail":"Not Found"}':
                logger.info("Readiness testing passed!")
                return server_process
        except subprocess.CalledProcessError as e:
            pass
        num_quota_retries -= 1
        logger.info(
            f"We're waiting for the LLM API to be up. Timeout after {num_quota_retries * check_interval} seconds.")
        time.sleep(check_interval)
    kill_server(server_process)
    return None


def build_llm_config_block(model_path, model_api_url, temperature):
    return {
        # "cache_seed": 42,  # disable cache to avoid disk cache errors
        "config_list": [
            {
                # "model": "Mistral-Nemo-Instruct-2407",
                "model": f"{model_path}",
                # "base_url": "http://localhost:5000/v1",
                "base_url": f"{model_api_url}",
                "api_type": "openai",  # "open_ai",
                "api_key": "sk-111111111111111",
                "price": [0.0, 0.0]
                # https://microsoft.github.io/autogen/0.2/docs/topics/llm-caching/

            }
        ],
        "temperature": temperature
    }
