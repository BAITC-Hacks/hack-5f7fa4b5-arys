import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from aml_graph.io import load_data
from aml_graph.validate import validate_data

def test_source_data():
    n,e,t=load_data(Path(__file__).parents[1]/'data'); validate_data(n,e,t)
    assert len(n)==2248 and len(e)==3119 and len(t)==4840
    assert int(n.is_seed.sum())==81
