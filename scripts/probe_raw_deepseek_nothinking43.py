"""Fixed paired DeepSeek judgments with thinking explicitly disabled.

Offline preparation and auditable request handling share the established
helper. This output tree never reuses any enabled-thinking experiment.
"""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import probe_raw_deepseek_thinking43 as runner

runner.ROOT_OUT = runner.base.RESEARCH / 'outputs/vrms_rag_probes/20261011_deepseek_nothinking43_01'
runner.THINKING_TYPE = 'disabled'
runner.REASONING_EFFORT = None
runner.OUTPUT_TOKENS = 8192
runner.API_TIMEOUT = 120
runner.OBJECTIVE_CHANGE = {
    'user_requested_thinking_disabled': True,
    'enabled_thinking_records_retained_as_separate_historical_experiment': True,
    'enabled_thinking_judgments_reused': False,
    'scope_reuse_policy': 'Only identical requests from this disabled-thinking errors43 tree may be reused for all146.'}

if __name__ == '__main__':
    runner.main()
