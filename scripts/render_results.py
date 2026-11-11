#!/usr/bin/env python3
"""Render aggregate experiment JSON as self-contained HTML and optional figures.

No raw prompts, responses, judgment reasons, or human audit rows are read.
Usage: .venv/bin/python scripts/render_results.py runs/experiments/development/fire/report.json --plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'hi-Latn': 'Hinglish',
             'ar': 'Arabic', 'sw': 'Swahili', 'jv': 'Javanese', 'th': 'Thai', 'ko': 'Korean', 'id': 'Indonesian'}
METRICS = {'content_safe': 'Content-safe on harmful prompts', 'degenerate': 'Degenerate answers',
           'harmful_refusal': 'Harmful full refusal', 'over_refusal': 'Benign over-refusal',
           'capability': 'Capability accuracy'}
LOWER_IS_BETTER = {'over_refusal', 'degenerate'}


def label(tag: str) -> str:
    original = tag.removesuffix('-q4').removesuffix('-nf4')
    if original == 'global':
        return 'Global'
    if original in ('fire', 'earth'):
        return original.title()
    return original.replace('merge_back', 'merge').replace('0p', '0.').removeprefix('fire-').removeprefix('earth-')


def measurements(report: dict) -> list[dict]:
    rows = []
    for tag, model in report['models'].items():
        for condition, langs in model.get('conditions', {}).items():
            for lang, metrics in langs.items():
                for metric in ('content_safe', 'degenerate', 'harmful_refusal', 'over_refusal'):
                    if metric in metrics:
                        cell = metrics[metric]
                        rows.append({'tag': tag, 'label': label(tag), 'condition': condition, 'lang': lang,
                                     'metric': metric, 'rate': cell['rate'], 'ci': cell['ci'], 'n': cell['n'],
                                     'unclear': cell.get('unclear'), 'partial': cell.get('partial')})
        for condition, langs in model.get('capability_by_condition', {}).items():
            for lang, cell in langs.items():
                if cell.get('accuracy') is not None:
                    rows.append({'tag': tag, 'label': label(tag), 'condition': condition, 'lang': lang,
                                 'metric': 'capability', 'rate': cell['accuracy'], 'ci': cell.get('ci95'),
                                 'n': cell['n']})
    return rows


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tiny Aya repair results</title>
<style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--foreground,CanvasText);font-family:var(--font-sans,system-ui,sans-serif)}
.note,.panel{padding:16px;border-radius:var(--radius,10px);background:var(--card,Canvas);border:1px solid var(--border,GrayText)}
.note{font-size:14px;line-height:1.55;margin-bottom:16px}.note strong{color:var(--warning,LinkText)}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}label{font-size:12px;color:var(--muted-foreground,GrayText);display:grid;gap:5px}
select{font:inherit;font-size:14px;max-width:100%;padding:8px 10px;border:1px solid var(--border,GrayText);border-radius:6px;background:var(--background,Canvas);color:var(--foreground,CanvasText)}
select:focus-visible{outline:2px solid var(--ring,Highlight);outline-offset:2px}
.axis{display:flex;justify-content:space-between;margin:0 44px 10px 132px;font-size:11px;color:var(--muted-foreground,GrayText)}
.group{margin:0 0 20px}.group:last-child{margin:0}.lang{font-size:14px;font-weight:600;margin-bottom:7px}
.row{display:grid;grid-template-columns:124px minmax(0,1fr) 40px;gap:8px;align-items:center;height:25px;font-size:12px}
.name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{position:relative;height:20px;border-left:1px solid var(--border,GrayText);background:linear-gradient(to right,transparent 49.7%,var(--border,GrayText) 50%,transparent 50.3%)}
.bar{position:absolute;left:0;top:5px;height:10px;background:var(--series);opacity:.72;border-radius:0 3px 3px 0}
.ci{position:absolute;top:9px;height:2px;background:var(--series)}.ci:before,.ci:after{content:'';position:absolute;width:2px;height:8px;top:-3px;background:var(--series)}.ci:before{left:0}.ci:after{right:0}
.value{text-align:right;font-variant-numeric:tabular-nums}.caption{color:var(--muted-foreground,GrayText);font-size:12px;line-height:1.6;margin:12px 0 0}
.empty{padding:20px 0;color:var(--muted-foreground,GrayText)}details{margin-top:16px;font-size:12px;line-height:1.6}summary{cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px 5px;text-align:left;border-bottom:1px solid var(--border,GrayText)}
@media(max-width:440px){.row{grid-template-columns:95px minmax(0,1fr) 38px;gap:5px;font-size:11px}.axis{margin-left:100px}.controls{gap:8px}select{font-size:12px;padding:7px}.panel{padding:12px}}
</style></head><body>
<div class="note"><strong id="status"></strong><div id="context"></div></div>
<div class="controls">
<label>Metric<select id="metric"><option value="content_safe">Content-safe ↑</option><option value="degenerate">Degenerate answers ↓</option><option value="harmful_refusal">Harmful full refusal ↑</option><option value="over_refusal">Benign over-refusal ↓</option><option value="capability">Capability accuracy ↑</option></select></label>
<label>System preamble<select id="condition"><option value="default">Default</option><option value="none">Removed</option></select></label>
<label>Weights<select id="precision"><option value="fp16">FP16 · all arms</option><option value="q4">Q4_K_M · selected repair</option><option value="nf4">NF4 · CUDA</option></select></label>
<label>Language<select id="language"><option value="all">All languages</option></select></label>
</div>
<div class="panel"><div class="axis"><span>0%</span><span>50%</span><span>100%</span></div><div id="chart" role="img"></div>
<p class="caption" id="caption"></p>
<details><summary>Counts and confidence intervals</summary><div style="overflow-x:auto"><table><thead><tr><th>Language</th><th>Model</th><th>Rate</th><th>95% CI</th><th>n</th></tr></thead><tbody id="cells"></tbody></table></div></details></div>
<p class="caption" id="limits"></p>
<script>
const data=__PAYLOAD__;
const $=id=>document.getElementById(id),percent=x=>(100*x).toFixed(1)+'%';
const colors=['var(--chart-1,LinkText)','var(--chart-2,VisitedText)','var(--chart-3,Highlight)','var(--chart-4,CanvasText)','var(--chart-5,ActiveText)','var(--chart-6,GrayText)'];
const tags=[...new Set(data.rows.map(r=>r.tag.replace(/-(q4|nf4)$/,'')))];
const precisionOf=tag=>tag.endsWith('-q4')?'q4':tag.endsWith('-nf4')?'nf4':'fp16';
for(const option of $('precision').options)option.disabled=!data.rows.some(r=>precisionOf(r.tag)===option.value);
$('status').textContent=data.profile==='development'?'Development sample · provisional':(data.validated?'Calibration verified':'Provisional · calibration incomplete');
const judges=data.judgeSet==='local'?'Local judges: Aya Expanse 8B and Qwen3-8B. ':data.judgeSet==='router'?'Router judge pair. ':'Rule signals only. ';
$('context').textContent=judges+(data.selected?'Repair selected on repair data: '+data.selected+'. ':'')+'Held-out results appear below.';
$('limits').textContent=data.success?'The selected repair passed the report’s configured acceptance checks.':'Empirical repair success has not been established. Translation review, human/router calibration, the full experiment, and CUDA NF4 are separate requirements.';
for(const lang of Object.keys(data.languages)){const o=document.createElement('option');o.value=lang;o.textContent=data.languages[lang];$('language').append(o)}
function render(){
const metric=$('metric').value,condition=$('condition').value,precision=$('precision').value,lang=$('language').value;
const rows=data.rows.filter(r=>r.metric===metric&&r.condition===condition&&precisionOf(r.tag)===precision&&(lang==='all'||r.lang===lang));
const chart=$('chart'),cells=$('cells');chart.replaceChildren();cells.replaceChildren();
chart.setAttribute('aria-label',data.metrics[metric]+' by language, '+condition+' preamble, '+precision+' weights');
for(const language of Object.keys(data.languages)){
const groupRows=rows.filter(r=>r.lang===language);if(!groupRows.length)continue;
const group=document.createElement('div');group.className='group';const heading=document.createElement('div');heading.className='lang';heading.textContent=data.languages[language];group.append(heading);
for(const r of groupRows){
const row=document.createElement('div');row.className='row';row.style.setProperty('--series',colors[tags.indexOf(r.tag.replace(/-(q4|nf4)$/,''))%colors.length]);
row.title=r.label+': '+percent(r.rate)+'; n='+r.n+(r.ci?'; 95% CI '+percent(r.ci[0])+'–'+percent(r.ci[1]):'')+(r.unclear!==undefined?'; unclear '+percent(r.unclear)+'; partial refusal '+percent(r.partial):'');
const name=document.createElement('span');name.className='name';name.textContent=r.label;const track=document.createElement('div');track.className='track';
const bar=document.createElement('span');bar.className='bar';bar.style.width=(100*r.rate)+'%';track.append(bar);
if(r.ci){const ci=document.createElement('span');ci.className='ci';ci.style.left=(100*r.ci[0])+'%';ci.style.width=(100*(r.ci[1]-r.ci[0]))+'%';track.append(ci)}
const value=document.createElement('span');value.className='value';value.textContent=Math.round(100*r.rate)+'%';row.append(name,track,value);group.append(row);
const tr=document.createElement('tr');for(const v of [data.languages[language],r.label,percent(r.rate),r.ci?percent(r.ci[0])+'–'+percent(r.ci[1]):'Unavailable',r.n]){const td=document.createElement('td');td.textContent=v;tr.append(td)}cells.append(tr);
}chart.append(group);
}
if(!rows.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No measurements available for this selection.';chart.append(empty)}
$('caption').textContent='Bars: '+data.metrics[metric]+'. Lines: 95% percentile-bootstrap intervals. '+(['over_refusal','degenerate'].includes(metric)?'Lower is better. ':'Higher is better. ')+(rows.length?'n = '+Math.min(...rows.map(r=>r.n))+'–'+Math.max(...rows.map(r=>r.n))+' per cell. ':'')+'Small samples and truncated answers limit interpretation; unanimous tiny samples can yield zero-width bootstrap intervals.';
}for(const id of ['metric','condition','precision','language'])$(id).addEventListener('change',render);render();
</script></body></html>'''


def plots(report: dict, rows: list[dict], directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    selected = report.get('acceptance_candidate')
    family = report['family']
    languages = list(dict.fromkeys(r['lang'] for r in rows))
    paths = []
    for precision, suffix in (('fp16', ''), ('q4', '-q4'), ('nf4', '-nf4')):
        tags = [t + suffix for t in ('global', family, selected) if t]
        if not any(r['tag'] in tags for r in rows):
            continue
        figure, axes = plt.subplots(2, len(METRICS), figsize=(4.7 * len(METRICS), 8), layout='constrained')
        for ri, condition in enumerate(('default', 'none')):
            for ci, (metric, title) in enumerate(METRICS.items()):
                ax = axes[ri, ci]
                plot_languages = [lang for lang in languages if any(
                    r['tag'] in tags and r['condition'] == condition and r['metric'] == metric
                    and r['lang'] == lang for r in rows)]
                for ti, tag in enumerate(tags):
                    x, values, low, high = [], [], [], []
                    for li, lang in enumerate(plot_languages):
                        cell = next((r for r in rows if (r['tag'], r['condition'], r['metric'], r['lang']) ==
                                     (tag, condition, metric, lang)), None)
                        if cell is None:
                            continue
                        x.append(li + (ti - 1) * .24)
                        values.append(100 * cell['rate'])
                        bounds = cell.get('ci') or [cell['rate'], cell['rate']]
                        low.append(max(0, 100 * (cell['rate'] - bounds[0])))
                        high.append(max(0, 100 * (bounds[1] - cell['rate'])))
                    ax.bar(x, values, width=.22, yerr=np.array([low, high]) if values else None,
                           capsize=2, color=('#287ea8', '#d97938', '#478961')[ti], label=label(tag), zorder=3)
                ax.set(ylim=(0, 105), yticks=(0, 25, 50, 75, 100), ylabel=f'Percent, preamble: {condition}',
                       title=title + (' ↓' if metric in LOWER_IS_BETTER else ' ↑'),
                       xticks=range(len(plot_languages)),
                       xticklabels=[LANGUAGES.get(l, l) for l in plot_languages])
                ax.tick_params(axis='x', labelrotation=25)
                ax.grid(axis='y', alpha=.2, zorder=0)
                ax.spines[['top', 'right']].set_visible(False)
        axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.3), ncol=3, frameon=False)
        figure.suptitle(f'Tiny Aya {family.title()} · {precision.upper()} · {report["profile"]} sample\n'
                        'Provisional local judgments; bars and 95% bootstrap intervals', fontsize=13)
        for extension in ('png', 'svg'):
            path = directory / f'report-{precision}.{extension}'
            figure.savefig(path, dpi=180, bbox_inches='tight')
            paths.append(path)
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--plots', action='store_true', help='Also save PNG/SVG figures; needs .[reporting]')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    rows = measurements(report)
    languages = list(dict.fromkeys(r['lang'] for r in rows))
    payload = {'rows': rows, 'profile': report['profile'], 'validated': report['validated'],
               'judgeSet': report.get('judge_set'), 'success': report.get('empirical_repair_success', False),
               'selected': report.get('acceptance_candidate'), 'metrics': METRICS,
               'languages': {l: LANGUAGES.get(l, l) for l in languages}}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', r'\u003c')
    output = args.report.with_suffix('.html')
    output.write_text(HTML.replace('__PAYLOAD__', encoded), encoding='utf-8')
    print(output)
    if args.plots:
        for path in plots(report, rows, args.report.parent):
            print(path)


if __name__ == '__main__':
    main()
