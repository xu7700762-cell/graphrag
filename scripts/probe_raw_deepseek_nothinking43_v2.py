"""Final fixed-temperature paired DeepSeek experiment, thinking disabled.

This isolated output never reuses the enabled or unfixed-temperature runs.
"""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import probe_raw_deepseek_nothinking43 as mode

runner = mode.runner
runner.ROOT_OUT = runner.base.RESEARCH / 'outputs/vrms_rag_probes/20261011_deepseek_nothinking43_02'
runner.TEMPERATURE = 0
runner.OBJECTIVE_CHANGE = {
    'user_requested_thinking_disabled': True,
    'temperature_fixed_to_original_baseline_zero': True,
    'earlier_enabled_and_unfixed_temperature_runs_retained_as_exploration': True,
    'earlier_exploration_judgments_reused': False,
    'scope_reuse_policy': 'Only identical requests from this disabled-thinking temperature-zero errors43 tree may be reused for all146.'}

if __name__ == '__main__':
    runner.main()
