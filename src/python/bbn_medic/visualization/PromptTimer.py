import time
import json

from bbn_medic.io.io_utils import fopen, JSONLGenerator

class PromptTimer:
    def __init__(self, outfile):
        self._start_time = None
        self._accum = dict()
        self._stage = None
        self.outfile = outfile

    def start(self, stage):
        self._accum[stage] = 0.0
        self._stage = stage
        self._start_time = time.perf_counter()

    def stop(self, stage):
        if stage != self._stage:
            self._stage = None
            self._start_time = None
            raise Exception("Stop stage does not match start stage"); 
        stop_time = time.perf_counter()
        self._accum[stage] = stop_time - self._start_time
        self._start_time = None

    def get_accum(self):
        accum = 0.0
        print("Accumulated times from the following stages:")
        for key in self._accum:
            print("  " + key)
            accum = accum + self._accum[key]
        return accum

    def count_prompts(self, data_file, jgen_class):
        data = jgen_class.read(data_file)
        prompts = set()
        for datum in data:
            prompts.add(datum.id)
        return len(prompts)

    def compute_times(self, prompts_file):
        dur = self.get_accum()
        num = self.count_prompts(prompts_file, JSONLGenerator)
        sec_per_prompt = dur/num
        prompts_per_sec = num/dur
        print(f"\nTotal Seconds: {round(dur, 1)}\nTotal Prompts: {num}\nSecsPerPrompt: {round(sec_per_prompt,2)}\nPromptsPerSec: {round(prompts_per_sec, 2)}")
        data = {
            'duration_in_sec': dur,
            'num_prompts': num,
            'sec_per_prompt': sec_per_prompt,
            'prompts_per_sec': prompts_per_sec
        }
        with fopen(self.outfile, 'w') as file:
            json_out = json.dumps(data)
            file.write(f"{json_out}\n")
