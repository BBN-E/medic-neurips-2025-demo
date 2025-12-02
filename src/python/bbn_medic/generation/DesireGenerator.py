import os
import sys

from tqdm.auto import tqdm

from bbn_medic.common.Desire import Desire
from bbn_medic.common.Disorder import Disorder
from bbn_medic.common.Prompt import Prompt
from bbn_medic.common.Style import Style
from bbn_medic.io.io_utils import StyleJSONLGenerator, JSONLGenerator, fopen
from bbn_medic.utils.text_utils import post_process_desire_text


class DesireGenerator:
    @staticmethod
    def generate(input_file, output_file, model_instance, num_desires, model_temp,
                 default_disorder,
                 styles_file=None,
                 desire_generation_prompts_file=None,
                 disorders_file=None,
                 max_generation_length=None,
                 use_default_prompt=False):

        # Styles
        styles = []
        if styles_file:
            for style_obj in StyleJSONLGenerator.read(styles_file):
                styles.append(style_obj)
        if len(styles) == 0:
            styles = [Style(id="dummy", text="standard")]

        # Desire generation prompts
        if use_default_prompt:
            desire_generation_prompt_objs = [Prompt(text=None, id="default_prompt", metadata={"default_prompt": True})]
        else:
            desire_generation_prompt_objs = []
        if desire_generation_prompts_file is not None:
            for obj in JSONLGenerator.read(desire_generation_prompts_file):
                desire_generation_prompt_objs.append(obj)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Write output file
        with fopen(output_file, "w") as g:

            # Disorders
            if disorders_file is not None:
                disorder_objs = []
                for obj in JSONLGenerator.read_all(disorders_file):
                    disorder_objs.append(obj)
            else:
                disorder_objs = [Disorder(text=default_disorder, id="default_disorder")]
            duplicate_desires = set()
            for disorder in disorder_objs:
                # Desires
                for obj in tqdm(JSONLGenerator.read_all(input_file), file=sys.stdout, desc="Overgenerating desires",
                                leave=False):
                    id = obj.id
                    source_text = obj.text

                    obj_type = obj.type
                    if obj_type != "Desire":
                        raise ValueError(f"Unsupported type {obj_type}")

                    generated_desire_objects = []
                    # add original desire to list if no disorders
                    if disorders_file is None:
                        generated_desire_objects.append(Desire(text=source_text, id=id))
                    for desire_generation_prompt_obj in desire_generation_prompt_objs:
                        for style_obj in styles:
                            prompt_style = style_obj.text
                            if desire_generation_prompt_obj.text is None:
                                if desire_generation_prompt_obj.metadata.get("default_prompt", False) is True:
                                    llm_prompt_text = model_instance.prompt_for_desire_to_desire(source_text,
                                                                                                 prompt_style)
                                else:
                                    raise ValueError(f"The given prompt has a null text field with no default")
                            else:
                                disorder_text = disorder.text
                                llm_prompt_text = model_instance.text_to_formatted_prompt(
                                    desire_generation_prompt_obj.text.format(source_text=source_text,
                                                                             disorder_text=disorder_text))
                            generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                                    num_return_sequences=num_desires)

                            generated_desires = [post_process_desire_text(x, max_length=max_generation_length) for x in
                                                 generated_text]
                            generated_desires = [x for x in generated_desires if x is not None]
                            for x in generated_desires:
                                x_lc = x.lower()
                                if x_lc in duplicate_desires:
                                    continue
                                metadata = {f"{obj_type}_metadata": obj.metadata, f"{obj_type}_source_id": id,
                                            "style_id": style_obj.id,
                                            "generation_prompt_id": desire_generation_prompt_obj.id}
                                if disorder is not None:
                                    metadata["disorder_id"] = disorder.id
                                metadata['desire_to_desire_generation_prompt'] = llm_prompt_text
                                generated_desire = Desire(text=x, metadata=metadata)
                                g.write(generated_desire.to_json() + "\n")
                                duplicate_desires.add(x_lc)
