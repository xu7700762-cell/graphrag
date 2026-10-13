"""Plot only recorded path calibration and measured tool costs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'outputs/vrms_v2/reports'
OUT=ROOT/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
for row,model in enumerate(('eegnet','eegconformer')):
    for col,split in enumerate(('validation','test')):
        report=json.loads((ROOT/f'evaluation_{model}_fixed_seed42_{split}.json').read_text(encoding='utf-8'))
        ax=axes[row,col]; ax.plot([0,1],[0,1],color='#999999',ls='--',lw=1,label='Ideal reference')
        for label,metrics,color,marker in [('Raw',report['raw_probability_0_5']['metrics'],'#bd7032','s'),('Temperature scaled',report['metrics'],'#237b8d','o')]:
            bins=[b for b in metrics['reliability_bins'] if b['count']]
            ax.scatter([b['mean_probability'] for b in bins],[b['observed_high_fraction'] for b in bins],
                       s=[25+12*b['count'] for b in bins],marker=marker,color=color,alpha=.8,
                       label=label)
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean predicted path probability',ylabel='Observed high-class fraction',
               title=f"{model} / {split} (accepted {report['metrics']['accepted_paths']}/{report['metrics']['total_paths']})\nRaw ECE={report['raw_probability_0_5']['metrics']['ECE_10_equal_width']:.3f}; scaled ECE={report['metrics']['ECE_10_equal_width']:.3f}")
        ax.grid(alpha=.15)
fig.suptitle('Path reliability: 10 equal-width bins; marker size shows bin count',fontsize=12)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=3)
for extension in ('png','svg'):fig.savefig(OUT/f'path_reliability.{extension}',dpi=200)
plt.close(fig)

available=[ROOT/f'strategy_comparison_{model}.json' for model in ('eegnet','eegconformer')]
if all(p.exists() for p in available):
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,path in zip(axes,available):
        report=json.loads(path.read_text(encoding='utf-8'));runs=report['reports'];x=np.arange(len(runs))
        a=[r['tool_counts'].get('BiomarkerCalculator',0) for r in runs]
        b=[r['tool_counts'].get('CovarianceAnalyzer',0) for r in runs]
        ax.bar(x-.18,a,.36,label='Spectrum',color='#237b8d');ax.bar(x+.18,b,.36,label='Covariance',color='#bd7032',hatch='//')
        ax.set_xticks(x, ['Deep only','Full evidence','Adaptive']);ax.set_ylabel('Actually executed auxiliary calls')
        ax.set_title(report['model_id']+'; path predictions identical');ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Accelerated replay: calls are measured, no accuracy improvement is claimed')
    for extension in ('png','svg'):fig.savefig(OUT/f'tool_usage.{extension}',dpi=200)
    plt.close(fig)
print(str(OUT))
