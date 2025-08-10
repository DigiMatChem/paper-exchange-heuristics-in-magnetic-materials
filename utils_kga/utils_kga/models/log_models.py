from datetime import datetime
import json


PATH_TO_COMPLETE_ID_LOG_MAPPING = "../results/complete_model_id_log_mapping.json"


def get_model_id(model_type: str, target_type: str, target_name: str,
                 data_filter: str, additional_info: str | None = None):
    if additional_info:
        model_id = "__".join(
            [datetime.today().strftime('%Y-%m-%d'), model_type, target_type, target_name, data_filter, additional_info])
    else:
        model_id = "__".join([datetime.today().strftime('%Y-%m-%d'), model_type, target_type, target_name, data_filter])
    with open(PATH_TO_COMPLETE_ID_LOG_MAPPING, "r") as f:
        id_log_map = json.load(f)
    if model_id in id_log_map:
        model_id = "__".join([model_id, datetime.today().strftime("%H-%M")])
    assert model_id not in id_log_map
    return model_id


def make_id_log_test_score_mapping(model_id: str,
                                   log_path: str,
                                   f1_test_weighted_or_minor_cv: float | None = None,
                                   f1_test_weighted_or_minor_ext: float | None = None,
                                   ) -> None:
    with open(PATH_TO_COMPLETE_ID_LOG_MAPPING, "r") as f:
        id_log_map = json.load(f)
    id_log_map[model_id] = {"log_path": log_path,
                            "f1_test_weighted_or_minor_cv": f1_test_weighted_or_minor_cv,
                            "f1_test_weighted_or_minor_ext": f1_test_weighted_or_minor_ext}
    with open(PATH_TO_COMPLETE_ID_LOG_MAPPING, "w") as f:
        json.dump(id_log_map, f)
