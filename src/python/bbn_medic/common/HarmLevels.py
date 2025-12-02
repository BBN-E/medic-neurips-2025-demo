class HarmLevels:
    def __init__(self):
        # Mapping of harm levels to numerical values: more serious harm corresponds to higher values

        self.mapping_of_harm_levels = {
            "none": 0,
            "very_low": 1,
            "very low": 1,
            "low": 2,
            "medium": 3,
            "high": 4,
            "very_high": 5,
            "very high": 5,
            "severe": 6,
            "life_threatening": 7,
            "life threatening": 7,
            "life-threatening": 7
        }

        # Inverse mapping of harm levels, from numerical values to strings

        self.inverse_mapping_of_harm_levels = {
            0: "none",
            1: "very_low",
            2: "low",
            3: "medium",
            4: "high",
            5: "very_high",
            6: "severe",
            7: "life_threatening"
        }

    def harm_string_to_harm_level(self, harm_string):
        return self.mapping_of_harm_levels[harm_string]

    def harm_level_to_harm_string(self, harm_level):
        return self.inverse_mapping_of_harm_levels[harm_level]

    def sorted_list_of_harm_strings(self, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = min(self.inverse_mapping_of_harm_levels.keys())
        if upper_bound is None:
            upper_bound = max(self.inverse_mapping_of_harm_levels.keys())
        return sorted([harm_string for harm_string in self.inverse_mapping_of_harm_levels.values()
                       if lower_bound <= self.mapping_of_harm_levels[harm_string] <= upper_bound], key=lambda x: self.mapping_of_harm_levels[x])
