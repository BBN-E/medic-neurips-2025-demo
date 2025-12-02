from bbn_medic.common.Patient import Patient


def test_patient_construct_and_patient_expression_generation():
    simulated_patient = {
        "patient_id": "testing",
        "age": 35,
        "gender": "male",
        "structured_symptoms": {
            "structured_symptoms": [
                {
                    "attribute_name": "BP",
                    "attribute_value": "120/60"
                },
                {
                    "attribute_name": "Temp",
                    "attribute_value": "37"
                }
            ]
        },
        "unstructured_symptoms": {
            "unstructured_symptoms": [
                {
                    "description": ["I have running nose.", "My nose is keeping running."]
                }
            ]
        },
        "allergies": {
            "category": "allergy",
            "items": [
                "dust"
            ],
        },
        "medications": {
            "category": "medication",
            "items": []
        },
        "medical_histories": {
            "category": "medical_history",
            "items": []
        },
        'last_ins_and_outs': {
            "unstructured_symptoms": []
        },
        "event": {
            "unstructured_symptoms": []
        }
    }
    patient = Patient(**simulated_patient)
    patient_expressions = sorted(patient.convert_to_complete_patient_descriptions(lang='en'), key=lambda x: len(x.text),
                                 reverse=True)

    assert len(patient_expressions) == 8
    patient_expression_texts = [i.text for i in patient_expressions]
    assert patient_expression_texts == [
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. My nose is keeping running. I'm allergic to dust. I don't take any medications. I have no concerning medical history.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. I have running nose. I'm allergic to dust. I don't take any medications. I have no concerning medical history.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. My nose is keeping running. I'm allergic to dust. I have no concerning medical history.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. I have running nose. I'm allergic to dust. I have no concerning medical history.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. My nose is keeping running. I'm allergic to dust. I don't take any medications.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. I have running nose. I'm allergic to dust. I don't take any medications.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. My nose is keeping running. I'm allergic to dust.",
        "I'm 35 years old. I'm a male. My BP is 120/60. My Temp is 37. I have running nose. I'm allergic to dust."]
