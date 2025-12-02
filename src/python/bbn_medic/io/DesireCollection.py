import copy

from bbn_medic.common.Desire import Desire
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import DesireJSONLGenerator, JSONLGenerator, PromptJSONLGenerator


class DesireCollection():
    DESIRE_SOURCE_ID_KEY = 'Desire_source_id'  # TODO these should probably be defined somewhere else
    PROMPT_SOURCE_ID_KEY = 'Prompt_source_id'  # TODO these should probably be defined somewhere else

    def __init__(self,
                 desires: list[Desire] | DesireJSONLGenerator = [],
                 prompts: list[Prompt] | PromptJSONLGenerator = [],
                 quiet_mode = False):
        """
        The DesireCollection is an object for amalgamating Desires and Prompts into their associated "lineages,"
        such that the connections between a source Desire, the Prompts it was used to generated, and the subsequence
        further Prompts derived from those prompts, is preserved and ready for visualization.

        The DesireCollection works across any number of jsonl files. This means it is possible for it to encounter
        a Prompt which was derived from another Prompt or Desire that it has not seen yet. To make the tool resilient
        for use, it stores these "orphan prompts" in a separate data structure until such time as their associated
        Desire and/or source Prompt are loaded.

        TODO: would be nice to have some idea of performance metrics (e.g.
        how long does it take to load up 100's or 1000's of jsonl files, or millions of lines?)

        TODO: list possible additional features:
        - get me all the manually-generated Desires and/or Prompts used as inputs to create this collection
        - get me all the artificially-generated Desires/Prompts contained in this collection
        - let it support more (or even infinitely?) recursive prompt id reference chains?

        Args:
            desires: list[Desire] | DesireJSONLGenerator - a list of desires to add to the collection
            prompts: list[Prompt] | PromptJSONLGenerator - a list of prompts to add to the collection
            quiet_mode = False)
        """

        # THESE ARE LIKE THE NODES IN OUR PROMPT GRAPH/TREE
        # top-level map of desire ids to desires
        self.desire_id_to_desire_dict: dict[str, Desire] = {}
        # top-level map of prompt ids to prompts
        self.prompt_id_to_prompt_dict: dict[str, Prompt] = {}

        # THESE ARE LIKE THE EDGES IN THE PROMPT GRAPH/TREE
        # track prompts derived from desires
        self.source_desire_id_to_prompt_id_set: dict[str, set[str]] = {}
        # track prompts derived from other prompts
        self.source_prompt_id_to_derived_prompt_id_set: dict[str, set[str]] = {}

        # governs whether or not to print informative non-error messages to stdout while processing inputs
        self.quiet_mode = quiet_mode

        self.add_desires(desires)
        self.add_prompts(prompts)

    def add_content_from_jsonl(self, items: JSONLGenerator):
        num_desires_added = 0
        num_prompts_added = 0
        for item in items:
            if type(item) is Prompt:
                self.add_prompts(item)
                num_prompts_added += 1
            elif type(item) is Desire:
                self.add_desires(item)
                num_desires_added += 1
            else:
                # TODO would be nice to include JSONL line number in errors like this
                print(f"Type {type(item)} not yet supported by DesireCollection. Skipping...")
        if not self.quiet_mode:
            print(f"Added {num_desires_added} Desires and {num_prompts_added} Prompts ")


    def remove_content_from_jsonl(self, items: JSONLGenerator):
        """ This method is useful when the DesireCollection is held by a service monitoring a directory for files
            being added and removed.
        """
        for item in items:
            if type(item) is Prompt:
                self.remove_prompts(item)
            elif type(item) is Desire:
                self.remove_desires(item)
            else:
                print(f"Type {type(item)} not yet supported by DesireCollection. Skipping...")

    def add_desires(self, desires: list[Desire] | DesireJSONLGenerator):
        # let this method take a single Desire object: if we received a single Desire object, stick it in a list
        if type(desires) is Desire:
            desires = [desires]

        for desire in desires:
            self.desire_id_to_desire_dict[desire.id] = desire
        return

    def add_prompts(self, prompts: Prompt | list[Prompt] | PromptJSONLGenerator):
        # if we received a single Prompt object, put in a list so the rest of the method operate on a list
        if type(prompts) is Prompt:
            prompts = [prompts]

        for prompt in prompts:
            if not self.quiet_mode:
                print(f"Already loaded prompt {prompt.id}; replacing")
            self.prompt_id_to_prompt_dict[prompt.id] = prompt

            source_desire_id = None
            source_prompt_id = None

            if self.DESIRE_SOURCE_ID_KEY in prompt.metadata:
                source_desire_id = prompt.metadata[self.DESIRE_SOURCE_ID_KEY]
                if source_desire_id not in self.source_desire_id_to_prompt_id_set:
                    self.source_desire_id_to_prompt_id_set[source_desire_id] = set()
                self.source_desire_id_to_prompt_id_set[source_desire_id].add(prompt.id)

            if self.PROMPT_SOURCE_ID_KEY in prompt.metadata:
                source_prompt_id = prompt.metadata[self.PROMPT_SOURCE_ID_KEY]
                if source_prompt_id not in self.source_prompt_id_to_derived_prompt_id_set:
                    self.source_prompt_id_to_derived_prompt_id_set[source_prompt_id] = set()
                self.source_prompt_id_to_derived_prompt_id_set[source_prompt_id].add(prompt.id)

            if source_desire_id is None and source_prompt_id is None:
                print(f"Malformed prompt {prompt.id} lacks a source desire or prompt id in its metadata. Skipping...")
                continue

    def remove_desires(self, desires: Desire | list[Desire] | DesireJSONLGenerator):
        # let this method take a single Desire object: if we received a single Desire object, stick it in a list
        if type(desires) is Desire:
            desires = list(desires)

        for desire in desires:
            del self.desires_dict[desire.id]

            # TODO delete any incoming edges pointing to this desire

    def remove_prompts(self, prompts: Prompt | list[Prompt] | PromptJSONLGenerator):
        # let this method take a single Desire object: if we received a single Desire object, stick it in a list
        if type(prompts) is Prompt:
            prompts = [prompts]

        for prompt in prompts:
            # if we HAVEN'T seen this prompt before, warn and move on
            if prompt.id not in self.prompt_id_to_prompt_dict:
                if not self.quiet_mode:
                    print(f"Prompt {prompt.id} has not been added to this collection. Skipping...")
                continue
            else:
                del self.prompt_id_to_prompt_dict[prompt.id]
                # TODO delete any incoming edges pointing to this prompt

    def get_desires(self) -> list[Desire]:
        """
        Builds a list of Desires containing all the prompts associated with those Desires loaded up until this point.
        If there are any Prompts associated with Desires which haven't been read into this collection, then they
        won't be included

        :return: list of Desire objects containing Prompts generated for them
        """
        # TODO might we worth caching this result for later rather than re-building it every time
        desires = []
        for desire_id in self.desire_id_to_desire_dict:
            desire = copy.deepcopy(self.desire_id_to_desire_dict[desire_id]) # make copy since we mutate (add prompts)
            if desire_id in self.source_desire_id_to_prompt_id_set:
                for prompt_id in self.source_desire_id_to_prompt_id_set[desire_id]:
                    prompt = self.prompt_id_to_prompt_dict[prompt_id]
                    desire.add_prompt(prompt)
                    if prompt_id in self.source_prompt_id_to_derived_prompt_id_set:
                        for derived_prompt_id in self.source_prompt_id_to_derived_prompt_id_set[prompt_id]:
                            derived_prompt = self.prompt_id_to_prompt_dict[derived_prompt_id]
                            desire.add_prompt(derived_prompt)
            desires.append(desire)
        return desires

    def get_prompts(self) -> list[Prompt]:
        prompts = []
        for prompt_id in self.prompt_id_to_prompt_dict:
            prompts.append(self.prompt_id_to_prompt_dict[prompt_id])
        return prompts