import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
from aml_graph.io import load_data
from aml_graph.validate import validate_data
from aml_graph.features import build_features
from aml_graph.scoring import score_roles,score_priority,add_evidence
from aml_graph.export import export_results

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--output',default='outputs'); p.add_argument('--config',default='config.json'); a=p.parse_args()
    cfg=json.load(open(a.config,encoding='utf-8')); nodes,edges,tx=load_data(a.data); validate_data(nodes,edges,tx)
    f,G,cmap=build_features(nodes,edges,tx,cfg['louvain_seed']); f=score_roles(f); f=score_priority(f,cfg['priority_weights'],cfg['seed_priority_multiplier'],cfg['isolated_priority_multiplier']); f=add_evidence(f)
    export_results(f,edges,cmap,a.output,cfg['top_n']); Path(a.output).mkdir(exist_ok=True); f.to_csv(Path(a.output)/'node_features.csv',index=False)
    print(f'OK: {len(f)} nodes, {len(edges)} edges, {len(tx)} transactions -> {a.output}')
if __name__=='__main__': main()
