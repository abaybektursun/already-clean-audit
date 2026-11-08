from __future__ import annotations

import copy
import pytest

from infra.vast.parallax.pg1_model_smoke import ModelSmokeError, validate_spec
from infra.vast.parallax.tests.test_s3_cache_hydration import hydration_spec


def smoke_spec() -> dict:
    d=hydration_spec(); d.update({
        'scope':'pg1_model_forward_smoke_only','uses_gpu':True,
        'gpu_work_class':'single_model_forward_smoke','scientific_claim':False,
        'synthetic_items':1,'discovery_items_read':0,'confirmation_items_read':0,
        'science_items_evaluated':0,'model_loads':1,'gpu_work':1,'model_timeout_seconds':180,
        'forward_contract':{'attention':'eager','use_cache':False,'dtype':'bfloat16','offline':True},
    }); return d


def test_model_smoke_is_one_synthetic_offline_forward_only():
    assert validate_spec(smoke_spec())['model_loads']==1


@pytest.mark.parametrize('field,value',[('synthetic_items',2),('confirmation_items_read',1),('scientific_claim',True),('model_loads',2)])
def test_model_smoke_rejects_scope_expansion(field,value):
    d=copy.deepcopy(smoke_spec()); d[field]=value
    with pytest.raises(ModelSmokeError): validate_spec(d)
