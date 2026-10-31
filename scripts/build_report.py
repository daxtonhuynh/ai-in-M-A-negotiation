from pathlib import Path
import csv, platform
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((R/'outputs/negotiation_batch_summary_by_method.csv').open()))
d={x['method_label']:x for x in rows}
styles=getSampleStyleSheet(); styles['BodyText'].leading=14
story=[]
def p(t):story.append(Paragraph(t,styles['BodyText']));story.append(Spacer(1,9))
def h(t):story.append(Paragraph(t,styles['Heading2']))
def table(data):
 t=Table(data,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17334a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f8'),colors.white])])) ;story.append(t)
story.append(Paragraph('AI for Merger Negotiation',styles['Title']))
p('A revised computational study of search efficiency and decision outcomes<br/>Portfolio edition | 31 October 2026 | Python '+platform.python_version())
h('Executive findings')
a=d['alpha_beta'];m=d['minimax']; reduction=100*(1-float(a['mean_nodes'])/float(m['mean_nodes']))
p(f"Across nine synthetic scenarios, ordered exact Alpha-Beta visited {float(a['mean_nodes']):,.2f} nodes on average versus {float(m['mean_nodes']):,.2f} for Minimax, a {reduction:.1f}% reduction in mean node visits. Both produced mean utility {float(m['mean_utility']):.3f} and agreement rate {float(m['mean_deal_success']):.1%}.")
p('Matched-depth experiments do not establish an outcome advantage for h3 over h1 or h2. At depth 3, all three have the same mean realized utility and absolute utility deviation from Minimax. Richer evaluation is therefore a hypothesis to investigate, not a demonstrated superiority claim.')
h('Purpose and scope')
p('This project translates a stylized merger bargaining problem into deterministic, fully observable, finite-horizon adversarial search. It compares exact search with approximate search under constrained depth. It is a methodological simulation using synthetic scenarios, not an empirically calibrated M&A decision system.')
h('Revision status')
p('This edition supersedes the previous report for the code and outputs in this repository. It corrects remaining-concession bounds, replaces the inactive feasibility-gap pressure proxy with an unresolved bid-ask pressure proxy, and adds matched-depth h1/h2 comparisons. Historical results must not be mixed with these revised results.')
story.append(PageBreak())
h('1. Model and implementation')
p('Buyer is MAX and seller is MIN. The state contains current turn, round, offer, reservation values, synergy, delay cost, each side\'s remaining concession budget and offer history. Legal actions are offer, accept and exit. Buyer offers are nondecreasing; seller offers are nonincreasing. Only overlapping feasible price zones are supported.')
p('A buyer\'s next upper offer is its previous bid plus its remaining budget, capped at buyer reservation plus synergy. A seller\'s next lower offer is its previous ask minus its remaining budget, floored at seller reservation. Each revision consumes only its incremental concession. This prevents the previous implementation from incorrectly shrinking movement relative to the original anchor.')
p('The round counter increments after a seller counteroffer. The horizon is terminal once round exceeds max_rounds; an offer at the final boundary can therefore expire before another response. This timing convention is part of the protocol, not a universal bargaining rule.')
h('2. Objective and evaluation')
p('Accepted-deal score = (buyer reservation + synergy - price) - (price - seller reservation) - delay cost. Default no-deal scoring remains asymmetric_relative: buyer exit is negative, seller exit positive before delay, and horizon failure has half the breakdown penalty. This is a relative adversarial score, not profit or joint welfare.')
p('h1 evaluates offer attractiveness. h2 adds elapsed delay cost. Revised h3 subtracts pressure when the unresolved ask-minus-bid gap exceeds combined remaining concessions or remains positive near the deadline. Missing bids/asks use initial aspiration anchors. Pressure is deterministic and is not a calibrated probability. h1/h2 remain deliberately approximate and do not exactly reproduce the terminal score.')
h('3. Experiment design')
p('Nine fixed scenarios: three buyer-advantaged, three balanced, three seller-advantaged. All use max_rounds=3. Compare Minimax; ordered/unordered exact Alpha-Beta; ordered h1, h2 and h3 at depth 2 and depth 3; and unordered h3 at depth 3. Both parties use the selected method in each simulated negotiation. Node counts are recursive visits, including terminal and cutoff calls, summed across decisions.')
p('Timing is one measured negotiation per scenario and method, with no repeated timing trials. Treat runtimes as illustrative. These short horizons and the small, confounded scenario groups do not identify causal effects of individual economic parameters.')
story.append(PageBreak())
h('4. Regenerated benchmark')
data=[['Method','Mean nodes','Utility','Deal rate','Abs. utility gap']]
for x in rows:
 data.append([x['method_label'],f"{float(x['mean_nodes']):,.1f}",f"{float(x['mean_utility']):.3f}",f"{float(x['mean_deal_success']):.1%}",f"{float(x['mean_absolute_utility_gap_vs_minimax']):.3f}"])
table(data)
p('Abs. utility gap is the scenario-level absolute realized-utility difference from Minimax, averaged across scenarios. It is a deviation measure, not exploitability or regret against an exact opponent.')
h('Interpretation')
p('Depth-2 methods achieve higher self-play scores and more agreements than exact search in this batch. This does not establish stronger strategies: both the buyer and seller use the approximate policy, so changed opponent behavior can improve the buyer score. A fixed-opponent or best-response evaluation is needed to measure strategic robustness.')
p('At depth 3, h1/h2/h3 yield mean utility -13.333 and mean absolute deviation 0.222. Their different node counts do not establish different outcome quality. Root-action mismatches may reflect ties; equal values are the primary exact-search correctness check.')
h('5. Validation')
p('Six automated tests passed: exact Minimax/Alpha-Beta value equality at all scenario roots and their immediate successors with ordering enabled/disabled; buyer and seller budget regression checks; rejection of illegal acceptance; positive pressure-penalty activation; and invalid offer-step rejection. This validates sampled behavior and specific regressions, not all possible states.')
story.append(PageBreak())
h('6. Search effort preview')
img=R/'outputs/plots/figure_02_nodes_by_method.png'
if img.exists():
 from PIL import Image as PI
 w,hh=PI.open(img).size;story.append(Image(str(img),width=475,height=475*hh/w))
h('7. Reproduction and next steps')
p('From repository root: install requirements.txt; run merger_negotiation_batch_experiments.py; run plot_negotiation_results.py; run python -m unittest discover -s tests -v. To recreate this PDF, install requirements-report.txt and run python scripts/build_report.py.')
p('Future work: broader scenarios and horizons, repeated timing trials, exact-opponent evaluation, payoff-mode sensitivity, and incomplete-information bargaining. The alternative symmetric_breakdown mode changes only the buyer-facing score; it is not a complete two-utility negotiation model.')
h('Evidence and provenance')
p('All numerical tables in this edition are generated directly from outputs/negotiation_batch_summary_by_method.csv. The three project modules are the implementation source. CHANGELOG.md documents changes from the supplied academic version. This report makes no literature-novelty claim and introduces no unverified external citations.')
def footer(c,doc):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526477'));c.drawString(45,25,'MERGER NEGOTIATION | Revised portfolio edition');c.drawRightString(550,25,str(doc.page))
SimpleDocTemplate(str(R/'reports/final-report.pdf'),rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
