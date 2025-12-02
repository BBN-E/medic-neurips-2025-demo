from datetime import datetime
import multiprocessing
import subprocess
import itertools
import os
import time

# adapted from https://gitlab.com/paloha/gpuMultiprocessing

SAFE_COLORS = [
    '\033[38;5;27m',   # blue
    '\033[38;5;34m',   # green
    '\033[38;5;202m',  # orange
    '\033[38;5;99m',   # purple
    '\033[38;5;160m',  # red
    '\033[38;5;45m',   # cyan
    '\033[38;5;208m',  # dark orange
    '\033[38;5;13m',   # magenta
    '\033[38;5;25m',   # navy
    '\033[38;5;64m',   # teal
    '\033[38;5;166m',  # burnt orange
    '\033[38;5;88m',   # dark red
    '\033[38;5;129m',  # violet
    '\033[38;5;33m',   # steel blue
    '\033[38;5;98m',   # plum
    '\033[38;5;136m',  # tan
    '\033[38;5;69m',   # sea blue
    '\033[38;5;149m',  # soft green
    '\033[38;5;173m',  # peach
    '\033[38;5;94m',   # muted purple
    '\033[38;5;37m',   # slate
    '\033[38;5;105m',  # dusty lavender
    '\033[38;5;214m',  # amber
    '\033[38;5;112m',  # fern green
    '\033[38;5;131m',  # mauve
    '\033[38;5;61m',   # deep blue
    '\033[38;5;171m',  # light plum
    '\033[38;5;181m',  # rose
    '\033[38;5;140m',  # muted orchid
    '\033[38;5;68m',   # marine
]

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '-L'], encoding='utf-8')
        gpu_info = output.strip()
        return gpu_info
    except FileNotFoundError:
        print("NVIDIA drivers are not installed or nvidia-smi is not in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")

def get_gpu_id(gpu_id_list=[0]):
    """
    This function is aimed to be run in a subprocess, e.g. by multiprocessing.Pool.
    It finds out which CPU is running the current process and based on its CPUid,
    it returns one GPUid. If this function is run in a MainProcess instead
    of a subprocess, the GPUid is always the first from the gpu_id_list.
    """

    # Get cpu id of current process
    cpu_name = multiprocessing.current_process().name
    try:
        cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    except:
        cpu_id = 0  # In case of MainProcess

    # Map cpu id to gpu id based on gpu_id_list
    gpu_id = gpu_id_list[cpu_id % len(gpu_id_list)]
    return gpu_id


def command_runner(command, gpu_id_list, env_gpu_name):
    """
    This function makes a subprocess.call to a specified command with the GPUid
    assigned to that process (based on its CPUid) as and ENV variable prepended
    to the command. The name of the ENV variable can be specified as well.
    Returns a tuple (return code, command)
    """

    # Get the GPUid of the current process
    current_process_gpu_id = get_gpu_id(gpu_id_list)

    # Make the GPU visible for the script using env variable
    env_gpu = '{}={}'.format(env_gpu_name, current_process_gpu_id)

    # Call the command with env_gpu variable prepended
    new_command = ' '.join([env_gpu, command])

    # Start the subprocess asynchronously
    process = subprocess.Popen(new_command, shell=True,
                               stdout=subprocess.PIPE,
                               bufsize=1,
                               text=True)

    proc = multiprocessing.current_process()

    while process.poll() is None:
        for line in process.stdout:
            color = SAFE_COLORS[hash(os.getpid()) % len(SAFE_COLORS)]
            reset = '\033[0m'
            print(f"{datetime.now().strftime('%H:%M:%S')} {color}{proc.name} ({current_process_gpu_id}): {line}{reset}", end='')
            
    return (process.returncode, command)


def queue_runner(command_queue, gpu_id_list, env_gpu_name='CUDA_VISIBLE_DEVICES',
                 processes_per_gpu=4, allowed_restarts=3,
                 command_runner_func=command_runner,
                 command_runner_args=[]):

    """
    This function uses starmap() from multiprocessing.Pool to run each command
    from the command queue with the command_runner_func while aware of the GPUid.

    This allows commands that can run on GPU to be run in parallel processes
    like in case of multiprocessing, but on all specified GPUs.


    Parameters:
    --------
    command_queue, list of strings
      Commands to be run with command_runner_func.
      E.g.: ['ENVVAR=1 python experiment.py', 'ENVVAR=2 python experiment.py']

    gpu_id_list, list of ints
      List of GPUids from which one can be assigned to the process.

    env_gpu_name, str, optional, default='CUDA_VISIBLE_DEVICES'
      Name of the environment variable in which the GPUid will be stored.

    processes_per_gpu, int, optional, default=4
      Allows to run multiple commands on one GPU at once.

    allowed_restarts, int, optional, default=3
      Restarts the multiprocessing.Pool if all the commands from the queue
      did not execute with 0 return code. Stops after allowed number of restarts.

    command_runner_func, callable, optional, default=command_runner
      A callable with signature containing (command, gpu_id_list) + arbitrary
      many args. This function will be run with starmap.

    command_runner_args, list, optional, default=[]
      Arguments for the command_runner_func.

    Returns the command_queue containing only the commands which failed with
    a non-zero exitcode. This can be for arbitrary reason, e.g. a bug in the
    script which is run by the command.
    """

    st = time.time()
    # Keeps track of how many times the pipe was restarted
    # If it is too many times
    restart_counter = 0
    while not len(command_queue) == 0:

        if restart_counter > allowed_restarts:
            if allowed_restarts > 0:  # let's only print this message if restarts are enabled
                print('Restart counter reached its limit. Some of the commands failed.')
            break
        if restart_counter > 0:
            print('Restarting {} failed commands.'.format(len(command_queue)))

        # Running commands in parallel on all available GPUs
        # If processes=len(gpu_id_list), all of the commands should succeed at the first go
        # If processes>len(gpu_id_list), the running process can take much less time,
        # but it is necessary to use this while loop in order to pick up interrupted
        # commands and run them again. Setting processes to reasonable number
        # should lead to not many commands to be interrupted. But it depends on lot of variables.

        with multiprocessing.Pool(processes=len(gpu_id_list) * processes_per_gpu) as pool:
            lq = len(command_queue)
            out = list(pool.starmap(command_runner_func,
                                    # Attributes for the command_runner
                                    zip(command_queue,
                                        itertools.repeat(gpu_id_list),
                                        itertools.repeat(env_gpu_name),
                                        *[itertools.repeat(a) for a in command_runner_args])))
            
            # Remove all commands from queue which succeeded
            command_queue = [x for ret_code, x in out if ret_code != 0]

            # print('Restarting {} failed commands.'.format(len(command_queue)))
            if lq == len(command_queue):
                # In this case, the queue len did not change since the last run
                # so that might indicate commands, that are not possible to run.
                # Keeping restart_counter to eventually kill the pipe no to loop
                # indifinetely.
                restart_counter += 1

    print('DONE. Total time: {:.4f} minutes.'.format((time.time() - st)/60))
    return command_queue

