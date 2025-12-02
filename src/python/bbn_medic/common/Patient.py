import itertools
import logging
import typing

import pydantic

from bbn_medic.utils.text_utils import get_id_from_text

logger = logging.getLogger("bbn_medic.common.Patient")


class PredefinedPatientExpressionTemplates:
    # TODO: remove empty case 
    en = {
        "age_gender": {
            "age_only": [],
            "gender_only": ["I'm a {gender}."],
        },
        "kv_pair": {
            "non_null": [
                "My {key} is {value}."
            ]
        },
        "allergy": {
            "null": [],
            "non_null": []
        },
        "medication": {
            "null": [], # fixed:
            "non_null": ["I'm taking {filler}."]
        },
        "medical_history": {
            "null": [],
            "non_null": ["I have a history of {filler}."]
        }
    }

    @staticmethod
    def get_templates(category: str, subtype: str, lang: str = 'en') -> typing.List[str]:
        if not hasattr(PredefinedPatientExpressionTemplates, lang):
            raise NotImplementedError(f"Unknown language {lang}")
        templates = getattr(PredefinedPatientExpressionTemplates, lang, {})
        ret = templates.get(category, {}).get(subtype, [])
        if len(ret) < 1:
            logger.warning(f"For category {category}, subtype {subtype}, there's no available template. ")
        return ret


class AtomicPatientExpression(pydantic.BaseModel):
    class Config:
        frozen = True

    id: str  # This is unique id when text string is different, the id should be different.
    meaning_id: str  # This is how we're phrasing a problem. When meaning of the text is the same, meaning id should be the same
    type: typing.Literal['AtomicPatientExpression'] = pydantic.Field(default="AtomicPatientExpression")
    style_id: str
    text: str
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(
        default={})  # This is used for calculate metrics, and later do the filtering.

    def strip_metadata(self):
        return AtomicPatientExpression(id=self.id, meaning_id=self.meaning_id, type=self.type, style_id=self.style_id,
                                       text=self.text, metadata=None)


class PatientExpression(pydantic.BaseModel):
    class Config:
        frozen = True

    type: typing.Literal['PatientExpression'] = pydantic.Field(default="PatientExpression")
    patient_id: str
    atomic_patient_expressions: typing.List[AtomicPatientExpression]
    is_complete: bool
    complete_meaning_id: str

    @property
    def text(self):
        return (" ".join(
            i.text for i in
            filter(lambda x: len(x.text.strip()) > 0, self.atomic_patient_expressions))).strip()

    @property
    def id(self):
        return "#".join(f"{i.id}" for i in self.atomic_patient_expressions)

    @property
    def meaning_id(self):
        return "#".join(i.meaning_id for i in self.atomic_patient_expressions)

    @property
    def style_id(self):
        all_possible_styles = set(i.style_id for i in self.atomic_patient_expressions)
        assert len(
            all_possible_styles) < 3, f"We expect it can have standard and modified, up to two styles. You gave {all_possible_styles}"
        if len(all_possible_styles) == 2:
            assert "standard" in all_possible_styles, f"Your style set is {all_possible_styles}"
            all_possible_styles.discard("standard")
            return list(all_possible_styles)[0]
        return list(all_possible_styles)[0]


class StructuredSymptom(pydantic.BaseModel):
    class Config:
        frozen = True

    attribute_name: str
    attribute_value: typing.Any

    def convert_to_descriptions(self, lang='en') -> typing.List[str]:
        templates = PredefinedPatientExpressionTemplates.get_templates('kv_pair', 'non_null', lang)
        return [t.format(key=self.attribute_name, value=self.attribute_value) for t in templates]


class StructuredSymptoms(pydantic.BaseModel):
    class Config:
        frozen = True

    structured_symptoms: typing.List[StructuredSymptom] = pydantic.Field(default_factory=list)

    def convert_to_descriptions(self, field_prefix, lang='en') -> typing.List[PatientExpression]:
        ret = []
        for idx, symptom in enumerate(self.structured_symptoms):
            symptom_id = get_id_from_text(f"{symptom.attribute_name}_{symptom.attribute_value}")
            texts = symptom.convert_to_descriptions(lang)
            local_buf = []
            for idx2, text in enumerate(texts):
                local_buf.append(AtomicPatientExpression(
                    meaning_id=f"{field_prefix}{symptom_id}",
                    id=f"{field_prefix}{symptom_id}|standard^{idx2}",
                    text=text,
                    style_id="standard"
                ))
            ret.append(local_buf)
        return ret


class GenericDescription(pydantic.BaseModel):
    class Config:
        frozen = True

    description: typing.List[str] = pydantic.Field(default_factory=list)

    def convert_to_descriptions(self, field_prefix, lang='en') -> typing.List[str]:
        return [i for i in self.description]


class UnstructuredSymptoms(pydantic.BaseModel):
    class Config:
        frozen = True

    unstructured_symptoms: typing.List[GenericDescription] = pydantic.Field(default_factory=list)

    def convert_to_descriptions(self, field_prefix, lang='en') -> typing.List[PatientExpression]:
        ret = []
        for idx, desc in enumerate(self.unstructured_symptoms):
            texts = desc.convert_to_descriptions(field_prefix, lang)
            phrase_id = get_id_from_text(texts[0])
            local_buf = []
            for idx2, text in enumerate(texts):
                local_buf.append(AtomicPatientExpression(
                    meaning_id=f"{field_prefix}{phrase_id}",
                    id=f"{field_prefix}{phrase_id}|standard^{idx2}",
                    text=text,
                    style_id="standard"
                ))
            ret.append(local_buf)
        return ret


class SimpleListToNarrativeModel(pydantic.BaseModel):
    class Config:
        frozen = True

    items: typing.List[str] = pydantic.Field(default_factory=list)
    category: str  # e.g., 'allergy', 'medication', etc.

    def convert_to_descriptions(self, field_prefix, lang='en') -> typing.List[PatientExpression]:
        ret = []
        if not self.items:
            local_buf = []
            for idx2, template in enumerate(
                    PredefinedPatientExpressionTemplates.get_templates(self.category, 'null', lang)):
                local_buf.append(AtomicPatientExpression(
                    meaning_id=f"{field_prefix}none",
                    id=f"{field_prefix}none|standard^{idx2}",
                    text=template,
                    style_id="standard"
                ))
            ret.append(local_buf)
        else:
            for idx, item in enumerate(self.items):
                phrase_id = get_id_from_text(item)
                local_buf = []
                for idx2, template in enumerate(
                        PredefinedPatientExpressionTemplates.get_templates(self.category, 'non_null', lang)):
                    local_buf.append(AtomicPatientExpression(
                        # patient_id=patient_id,
                        meaning_id=f"{field_prefix}{phrase_id}",
                        id=f"{field_prefix}{phrase_id}|standard^{idx2}",
                        text=template.format(filler=item),
                        style_id="standard"
                    ))
                ret.append(local_buf)
        return ret


class Patient(pydantic.BaseModel):
    """
    We assume this is the container all properties that a human being can have
    """
    type: typing.Literal['Patient'] = pydantic.Field(default="Patient")
    patient_id: str
    # Demographic
    age: typing.Optional[int] = pydantic.Field(None)
    gender: typing.Optional[str] = pydantic.Field(None)
    race: typing.Optional[typing.List[str]] = pydantic.Field(
        None)  # This is fully controlled at external. The list looks like
    # ["I'm Black.", "I'm African American."]

    # https://www.ems1.com/ems-products/epcr-electronic-patient-care-reporting/articles/how-to-use-sample-history-as-an-effective-patient-assessment-tool-J6zeq7gHyFpijIat/

    structured_symptoms: StructuredSymptoms = pydantic.Field(default_factory=StructuredSymptoms)
    unstructured_symptoms: UnstructuredSymptoms = pydantic.Field(default_factory=UnstructuredSymptoms)
    allergies: SimpleListToNarrativeModel = pydantic.Field(
        default_factory=lambda: SimpleListToNarrativeModel(category='allergy'))
    medications: SimpleListToNarrativeModel = pydantic.Field(
        default_factory=lambda: SimpleListToNarrativeModel(category='medication'))
    medical_histories: SimpleListToNarrativeModel = pydantic.Field(
        default_factory=lambda: SimpleListToNarrativeModel(category='medical_history'))
    last_ins_and_outs: UnstructuredSymptoms = pydantic.Field(default_factory=UnstructuredSymptoms)
    event: UnstructuredSymptoms = pydantic.Field(default_factory=UnstructuredSymptoms)

    @staticmethod
    def dfs_generate_patient_descriptions(string_combos, patient_id, current_cur, cur_buf, ret_buf,
                                          must_be_completed=False):
        if current_cur == len(string_combos):
            dummy_patient_expression = PatientExpression(patient_id=patient_id, atomic_patient_expressions=cur_buf,
                                                         is_complete=True, complete_meaning_id="")
            complete_phrase_id = dummy_patient_expression.meaning_id
            ret_buf.append(
                PatientExpression(patient_id=patient_id, atomic_patient_expressions=list(cur_buf), is_complete=True,
                                  complete_meaning_id=complete_phrase_id))
        else:
            # Option 1, bypass current condition
            if must_be_completed is False:
                Patient.dfs_generate_patient_descriptions(string_combos, patient_id, current_cur + 1, cur_buf, ret_buf,
                                                          must_be_completed=must_be_completed)
            # Option 2, take elements from current condition
            phrase_id_to_patient_description_id_narrative = dict()
            for patient_description_id_narrative in string_combos[current_cur]:
                phrase_id_to_patient_description_id_narrative.setdefault(patient_description_id_narrative.meaning_id,
                                                                         []).append(patient_description_id_narrative)
            fixed_phrase_id_list = list(sorted(phrase_id_to_patient_description_id_narrative.keys()))
            for n_situation in range(1 if must_be_completed is False else len(fixed_phrase_id_list),
                                     len(fixed_phrase_id_list) + 1):
                # "situation here means atomic description of a uniquely identifiable event/status. Such as I ate an ice cream today. I ate an apple today. These two are two different situation."
                for combo_phrase_ids in itertools.combinations(fixed_phrase_id_list, n_situation):
                    buckets = [phrase_id_to_patient_description_id_narrative[i] for i in combo_phrase_ids]
                    # For each chosen "situation", we'd like to include all its possible phases
                    # Situation her means a group of phases saying the same thing. For instance: ["I don't take any medications.", "I'm not taking any medications."]
                    for phases_of_different_situation in itertools.product(*buckets):
                        phrase_id = "_".join(i.meaning_id for i in phases_of_different_situation)
                        narrative_string_id = "_".join(i.id for i in phases_of_different_situation)
                        narrative = "".join(i.text for i in phases_of_different_situation)
                        cur_buf.append(AtomicPatientExpression(
                            # patient_id=patient_id,
                            meaning_id=phrase_id,
                            id=narrative_string_id,
                            text=narrative,
                            style_id="standard"))
                        Patient.dfs_generate_patient_descriptions(string_combos, patient_id, current_cur + 1,
                                                                  cur_buf, ret_buf,
                                                                  must_be_completed=must_be_completed)
                        cur_buf.pop()

    def convert_to_complete_patient_descriptions(self, lang='en') -> typing.List[PatientExpression]:
        if lang == 'en':
            age_gender_descriptions = []
            if self.age is not None:
                for idx, template in enumerate(
                        getattr(PredefinedPatientExpressionTemplates, lang)['age_gender']['age_only']):
                    age_gender_descriptions.append([
                        AtomicPatientExpression(
                            meaning_id=f"age_{self.age}", id=f"age_{self.age}|standard^{idx}",
                            text=template.format(age=self.age),
                            style_id="standard")])
            if self.gender is not None:
                for idx, template in enumerate(
                        getattr(PredefinedPatientExpressionTemplates, lang)['age_gender']['gender_only']):
                    age_gender_descriptions.append([
                        AtomicPatientExpression(
                            meaning_id=f"gender_{self.gender}", id=f"gender_{self.gender}|standard^{idx}",
                            text=template.format(gender=self.gender),
                            style_id="standard")])
            if self.race is not None:
                local_rephrasing = []
                race_id = get_id_from_text(self.race[0])
                for idx, race_phrase in enumerate(self.race):
                    local_rephrasing.append(
                        AtomicPatientExpression(
                            meaning_id=f"race_{race_id}", id=f"race_{race_id}|standard^{idx}",
                            text=race_phrase.format(race=race_id),
                            style_id="standard"
                        )
                    )
                age_gender_descriptions.append(local_rephrasing)

            string_combos = []
            string_combos.extend(age_gender_descriptions)
            # Each of this is an array where each element of the array is a possible patient description of the property
            for prefix, collection in [
                ("structured_symptom_", self.structured_symptoms),
                ("unstructured_symptom_", self.unstructured_symptoms),
                ("allergy_", self.allergies),
                ("medication_", self.medications),
                ("medical_history_", self.medical_histories),
                ("last_in_and_out_", self.last_ins_and_outs),
                ("event_", self.event)
            ]:
                descs = collection.convert_to_descriptions(prefix, lang=lang)
                string_combos.extend(descs)
            string_combos = list(filter(lambda x: len(x) > 0, string_combos))

            theory_cnt = 1
            for combo in string_combos:
                phase_id_to_narratives = {}
                for i in combo:
                    phase_id_to_narratives.setdefault(i.meaning_id, list()).append(i)
                local_n_case = 0
                for phase_id, narratives in phase_id_to_narratives.items():
                    local_n_case += len(narratives)
                theory_cnt *= local_n_case
            logger.info(f"We're expecting to have {theory_cnt} narratives.")

            ret_buf = []
            Patient.dfs_generate_patient_descriptions(string_combos, self.patient_id, 0, [], ret_buf,
                                                      must_be_completed=True)
            logger.info(f"Before deduplication, we have {len(ret_buf)} narratives.")
            narrative_to_narrative_objs = dict()
            for i in ret_buf:
                narrative_to_narrative_objs.setdefault(i.text, []).append(i)
            deduplicated_buf = list()
            for narrative_text, narrative_objs in narrative_to_narrative_objs.items():
                if len(narrative_text) > 0:
                    deduplicated_buf.append(
                        sorted(narrative_objs, key=lambda x: len(x.atomic_patient_expressions))[
                            0])  # We want the simplest case
            logger.info(f"After deduplication, we have {len(deduplicated_buf)} narratives.")
            return deduplicated_buf

        else:
            raise NotImplementedError(f"{lang} is not implemented.")
