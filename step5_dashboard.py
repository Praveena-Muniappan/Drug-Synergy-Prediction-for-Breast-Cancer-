"""
=============================================================
 Phase 2 — Step 5: 3D Dashboard with Synergy Predictions
=============================================================
Combines:
  - 3D conformer viewer (3Dmol.js)
  - Predicted synergy heatmap across all 39 cell lines
  - Drug info panel
  - ALL drug pair combinations from the dataset

Click-to-expand accordion UI.
Auto-opens in browser on completion.
=============================================================
"""

import os, sys, pickle, warnings, webbrowser, json
import numpy as np
import pandas as pd
import collections
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Descriptors3D

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_predict_synergy import predict, load_resources

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DATASET     = "/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl"
OUTPUT_HTML = os.path.join(BASE_DIR, "synergy_dashboard.html")

# ── Drug info palette ──────────────────────────────────────────────────────────
DRUG_INFO = {
    "5-FU":             {"full_name":"5-Fluorouracil",       "drug_class":"Antimetabolite",            "color":"#ff6b6b"},
    "ABT-888":          {"full_name":"Veliparib",            "drug_class":"PARP Inhibitor",             "color":"#ffa94d"},
    "AZD1775":          {"full_name":"Adavosertib",          "drug_class":"WEE1 Kinase Inhibitor",      "color":"#69db7c"},
    "BEZ-235":          {"full_name":"Dactolisib",           "drug_class":"PI3K/mTOR Inhibitor",        "color":"#74c0fc"},
    "BORTEZOMIB":       {"full_name":"Bortezomib",           "drug_class":"Proteasome Inhibitor",       "color":"#da77f2"},
    "CARBOPLATIN":      {"full_name":"Carboplatin",          "drug_class":"Platinum Agent",             "color":"#a9e34b"},
    "CYCLOPHOSPHAMIDE": {"full_name":"Cyclophosphamide",     "drug_class":"Alkylating Agent",           "color":"#ff8787"},
    "DASATINIB":        {"full_name":"Dasatinib",            "drug_class":"BCR-ABL/Src TKI",            "color":"#ffd43b"},
    "DEXAMETHASONE":    {"full_name":"Dexamethasone",        "drug_class":"Glucocorticoid",             "color":"#63e6be"},
    "DINACICLIB":       {"full_name":"Dinaciclib",           "drug_class":"CDK1/2/5/9 Inhibitor",      "color":"#91a7ff"},
    "DOXORUBICIN":      {"full_name":"Doxorubicin",          "drug_class":"Anthracycline",              "color":"#f06595"},
    "ERLOTINIB":        {"full_name":"Erlotinib",            "drug_class":"EGFR TKI",                   "color":"#ffc9c9"},
    "ETOPOSIDE":        {"full_name":"Etoposide",            "drug_class":"Topoisomerase II Inhibitor", "color":"#b2f2bb"},
    "GELDANAMYCIN":     {"full_name":"Geldanamycin",         "drug_class":"HSP90 Inhibitor",            "color":"#ffec99"},
    "GEMCITABINE":      {"full_name":"Gemcitabine",          "drug_class":"Antimetabolite",             "color":"#a5d8ff"},
    "L778123":          {"full_name":"L778123",              "drug_class":"Farnesyltransferase Inhib.", "color":"#d0bfff"},
    "LAPATINIB":        {"full_name":"Lapatinib",            "drug_class":"EGFR/HER2 TKI",             "color":"#ffa8a8"},
    "METFORMIN":        {"full_name":"Metformin",            "drug_class":"AMPK Activator",             "color":"#96f2d7"},
    "METHOTREXATE":     {"full_name":"Methotrexate",         "drug_class":"Antimetabolite",             "color":"#74c0fc"},
    "MITOMYCINE":       {"full_name":"Mitomycin C",          "drug_class":"Alkylating Agent",           "color":"#e599f7"},
    "MK-2206":          {"full_name":"MK-2206",              "drug_class":"AKT Inhibitor",              "color":"#ffc078"},
    "MK-4541":          {"full_name":"MK-4541",              "drug_class":"AR Antagonist",              "color":"#66d9e8"},
    "MK-4827":          {"full_name":"Niraparib",            "drug_class":"PARP Inhibitor",             "color":"#a9e34b"},
    "MK-5108":          {"full_name":"MK-5108",              "drug_class":"Aurora A Inhibitor",         "color":"#ff8787"},
    "MK-8669":          {"full_name":"Ridaforolimus",        "drug_class":"mTOR Inhibitor",             "color":"#ffd43b"},
    "MK-8776":          {"full_name":"MK-8776",              "drug_class":"CHK1 Inhibitor",             "color":"#63e6be"},
    "MRK-003":          {"full_name":"MRK-003",              "drug_class":"Gamma-Secretase Inhibitor",  "color":"#91a7ff"},
    "OXALIPLATIN":      {"full_name":"Oxaliplatin",          "drug_class":"Platinum Agent",             "color":"#f06595"},
    "PACLITAXEL":       {"full_name":"Paclitaxel",           "drug_class":"Taxane",                     "color":"#ffc9c9"},
    "PD325901":         {"full_name":"PD325901",             "drug_class":"MEK Inhibitor",              "color":"#b2f2bb"},
    "SN-38":            {"full_name":"SN-38",                "drug_class":"Topoisomerase I Inhibitor",  "color":"#ffec99"},
    "SORAFENIB":        {"full_name":"Sorafenib",            "drug_class":"Multikinase Inhibitor",      "color":"#a5d8ff"},
    "SUNITINIB":        {"full_name":"Sunitinib",            "drug_class":"VEGFR/PDGFR TKI",           "color":"#d0bfff"},
    "TEMOZOLOMIDE":     {"full_name":"Temozolomide",         "drug_class":"Alkylating Agent",           "color":"#ffa8a8"},
    "TOPOTECAN":        {"full_name":"Topotecan",            "drug_class":"Topoisomerase I Inhibitor",  "color":"#96f2d7"},
    "VINBLASTINE":      {"full_name":"Vinblastine",          "drug_class":"Vinca Alkaloid",             "color":"#74c0fc"},
    "VINORELBINE":      {"full_name":"Vinorelbine",          "drug_class":"Vinca Alkaloid",             "color":"#e599f7"},
    "ZOLINZA":          {"full_name":"Vorinostat",           "drug_class":"HDAC Inhibitor",             "color":"#ffc078"},
}

ATOM_COLORS = {
    "C":"#909090","H":"#ffffff","N":"#3050f8","O":"#ff0d0d",
    "F":"#90e050","Cl":"#1ff01f","S":"#ffff30","P":"#ff8000",
    "B":"#ffb5b5","Pt":"#d0d0e0"
}

# ── Load resources ──────────────────────────────────────────────────────────────
print("=" * 65)
print(" Phase 2 — Step 5: 3D Dashboard (ALL Combinations)")
print("=" * 65)

print("\nLoading model and features...")
drug_features, cell_embeddings, bundle = load_resources()
print(f"  Model      : {bundle['model_name']}")
print(f"  Drugs      : {len(drug_features)}")
print(f"  Cell lines : {len(cell_embeddings)}")

df = pd.read_pickle(DATASET)
smiles_map = dict(zip(df["Drug1_ID"], df["Drug1"]))
smiles_map.update(dict(zip(df["Drug2_ID"], df["Drug2"])))

# ── Get ALL unique drug pairs ──────────────────────────────────────────────────
all_pairs = (df[["Drug1_ID","Drug2_ID"]]
             .drop_duplicates()
             .reset_index(drop=True))
all_pairs = all_pairs[
    all_pairs["Drug1_ID"].isin(drug_features) &
    all_pairs["Drug2_ID"].isin(drug_features)
].reset_index(drop=True)

print(f"\nTotal unique drug pairs: {len(all_pairs)}")

# ── Pre-compute 3D SDF blocks for each unique drug ──────────────────────────────
print("\nBuilding 3D conformers for all unique drugs...")
unique_drug_ids = set(all_pairs["Drug1_ID"]) | set(all_pairs["Drug2_ID"])
sdf_map   = {}
mol_props = {}

for drug_id in sorted(unique_drug_ids):
    smiles = smiles_map.get(drug_id, "")
    if not smiles:
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    formula     = rdMolDescriptors.CalcMolFormula(mol)
    mw          = round(Descriptors.MolWt(mol), 2)
    logp        = round(Descriptors.MolLogP(mol), 2)
    hbd         = rdMolDescriptors.CalcNumHBD(mol)
    hba         = rdMolDescriptors.CalcNumHBA(mol)
    rings       = rdMolDescriptors.CalcNumRings(mol)
    atom_counts = collections.Counter(a.GetSymbol() for a in mol.GetAtoms())
    total_h     = sum(atom_counts.values())
    atom_comp   = {s:(c, round(c/total_h*100, 1))
                   for s,c in sorted(atom_counts.items(), key=lambda x:-x[1])}
    mol_props[drug_id] = dict(
        formula=formula, mw=mw, logp=logp,
        hbd=hbd, hba=hba, rings=rings, atom_comp=atom_comp
    )

    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, params) != 0:
        continue
    AllChem.MMFFOptimizeMolecule(mol3d)
    sdf_map[drug_id] = Chem.MolToMolBlock(mol3d)
    print(f"  3D ok: {drug_id}")

# ── Predict synergy for EVERY pair ──────────────────────────────────────────────
print(f"\nPredicting synergy for {len(all_pairs)} pairs ...")
pair_data = []

for idx, row in all_pairs.iterrows():
    da, db = row["Drug1_ID"], row["Drug2_ID"]
    if da not in sdf_map or db not in sdf_map:
        continue
    try:
        syn_df = predict(
            smiles_a=smiles_map.get(da,""),
            smiles_b=smiles_map.get(db,""),
            cell_line_id="all",
            drug_id_a=da, drug_id_b=db,
            drug_features=drug_features,
            cell_embeddings=cell_embeddings,
            bundle=bundle,
        )
    except Exception as e:
        print(f"  [warn] {da}+{db}: {e}")
        syn_df = None

    avg_syn, max_syn, top_cl, synergy_rows = None, None, None, []
    if syn_df is not None and not syn_df.empty:
        sd = syn_df.dropna()
        avg_syn = round(float(sd["predicted_synergy"].mean()), 2)
        max_syn = round(float(sd["predicted_synergy"].max()), 2)
        top_cl  = sd.loc[sd["predicted_synergy"].idxmax(), "cell_line"]
        for _, r in sd.sort_values("predicted_synergy", ascending=False).iterrows():
            synergy_rows.append({
                "cl":    r["cell_line"],
                "val":   round(float(r["predicted_synergy"]), 2),
                "interp":r["interpretation"]
            })

    def sc(v):
        if v is None: return "neutral"
        if v > 10:    return "strong"
        if v > 0:     return "mild"
        if v > -10:   return "neutral"
        return "antagonist"

    info_a = DRUG_INFO.get(da, {})
    info_b = DRUG_INFO.get(db, {})
    pair_data.append({
        "id":          f"{da}__{db}".replace("-","_").replace("+","plus").replace(" ","_"),
        "drug_a": da, "drug_b": db,
        "fn_a":   info_a.get("full_name",  da),
        "fn_b":   info_b.get("full_name",  db),
        "cls_a":  info_a.get("drug_class", "—"),
        "cls_b":  info_b.get("drug_class", "—"),
        "col_a":  info_a.get("color",      "#6c63ff"),
        "col_b":  info_b.get("color",      "#00d4ff"),
        "sdf_a":  sdf_map.get(da, ""),
        "sdf_b":  sdf_map.get(db, ""),
        "props_a":mol_props.get(da, {}),
        "props_b":mol_props.get(db, {}),
        "avg_syn":avg_syn, "max_syn":max_syn, "top_cl":top_cl,
        "synergy_rows":synergy_rows,
        "sc":     sc(avg_syn),
    })
    print(f"  [{idx+1}/{len(all_pairs)}] {da} + {db}  avg={avg_syn}")

print(f"\nDone. {len(pair_data)} pairs built.")

# ── HTML helpers ─────────────────────────────────────────────────────────────────
def atom_bar_html(atom_comp):
    bars, legend = [], []
    for sym,(cnt,pct) in atom_comp.items():
        c = ATOM_COLORS.get(sym,"#aaa")
        bars.append(f'<div style="width:{pct}%;background:{c};height:100%;display:inline-block;" title="{sym}:{pct}%"></div>')
        legend.append(f'<span class="comp-tag" style="border-color:{c}40;color:{c};">'
                      f'<span class="comp-dot" style="background:{c};"></span>{sym} {cnt}</span>')
    return (f'<div class="comp-bar">{"".join(bars)}</div>'
            f'<div class="comp-legend">{"".join(legend)}</div>')


def props_html(props, drug_id, fn, color):
    if not props:
        return f'<div class="no-data">No properties for {drug_id}</div>'
    ac = atom_bar_html(props.get("atom_comp", {}))
    return f"""
<div class="prop-card">
  <div class="prop-card-name" style="color:{color};">{fn}</div>
  <div class="prop-rows">
    <div class="prop-row"><span class="prop-lbl">Formula</span><span class="prop-val">{props.get('formula','—')}</span></div>
    <div class="prop-row"><span class="prop-lbl">Mol. Weight</span><span class="prop-val">{props.get('mw','—')} g/mol</span></div>
    <div class="prop-row"><span class="prop-lbl">LogP</span><span class="prop-val">{props.get('logp','—')}</span></div>
    <div class="prop-row"><span class="prop-lbl">HB Donors / Acceptors</span><span class="prop-val">{props.get('hbd','—')} / {props.get('hba','—')}</span></div>
    <div class="prop-row"><span class="prop-lbl">Rings</span><span class="prop-val">{props.get('rings','—')}</span></div>
  </div>
  <hr class="prop-divider"/>
  {ac}
</div>"""


def make_pair_row(p):
    pid      = p["id"]
    avg      = p["avg_syn"]
    max_s    = p["max_syn"]
    top_cl   = p["top_cl"] or "—"
    sc       = p["sc"]
    avg_disp = f"{avg:+.2f}" if avg is not None else "—"
    max_disp = f"{max_s:+.2f}" if max_s is not None else "—"
    syn_json = json.dumps(p["synergy_rows"])
    sdf_a    = (p["sdf_a"] or "").replace("`","\\`").replace("${","\\${")
    sdf_b    = (p["sdf_b"] or "").replace("`","\\`").replace("${","\\${")
    col_a, col_b = p["col_a"], p["col_b"]
    pa_html  = props_html(p["props_a"], p["drug_a"], p["fn_a"], col_a)
    pb_html  = props_html(p["props_b"], p["drug_b"], p["fn_b"], col_b)

    sc_css   = {"strong":"sc-s","mild":"sc-m","neutral":"sc-n","antagonist":"sc-a"}.get(sc,"sc-n")
    pill_css = {"strong":"pill-s","mild":"pill-m","neutral":"pill-n","antagonist":"pill-a"}.get(sc,"pill-n")
    sc_label = {"strong":"Strong Synergy","mild":"Mild Synergy","neutral":"Neutral","antagonist":"Antagonistic"}.get(sc,sc)

    return f"""
<div class="pc" id="pair-{pid}"
     data-da="{p['drug_a']}" data-db="{p['drug_b']}"
     data-fa="{p['fn_a']}"  data-fb="{p['fn_b']}"
     data-ca="{p['cls_a']}" data-cb="{p['cls_b']}"
     data-avg="{avg if avg is not None else ''}" data-sc="{sc}">

  <div class="pc-row" onclick="togglePair('{pid}')">
    <div>
      <div class="pc-drugs">
        <span class="drug-tag" style="--dc:{col_a};">{p['drug_a']}</span>
        <span class="plus-sep">+</span>
        <span class="drug-tag" style="--dc:{col_b};">{p['drug_b']}</span>
      </div>
      <div class="pc-sub">{p['fn_a']} &amp; {p['fn_b']} &nbsp;&middot;&nbsp; {p['cls_a']} &times; {p['cls_b']}</div>
    </div>

    <div class="pc-score">
      <span class="score-num {sc_css}">{avg_disp}</span>
      <span class="score-lbl">avg synergy</span>
    </div>

    <span class="sc-pill {pill_css}">{sc_label}</span>

    <svg class="caret" id="icon-{pid}" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  </div>

  <div class="pc-body" id="body-{pid}">

    <div class="viewers-grid">
      <div class="vp">
        <div class="vp-head">
          <span class="vp-name" style="color:{col_a};">{p['drug_a']}</span>
          <span class="vp-full">{p['fn_a']}</span>
          <span class="vp-cls">{p['cls_a']}</span>
        </div>
        <div id="v3d-a-{pid}" class="mol-viewer"></div>
        <div class="view-btns" id="sbA-{pid}">
          <button class="vbtn active" onclick="setStyleA_{pid}('ballstick',this)">Ball+Stick</button>
          <button class="vbtn" onclick="setStyleA_{pid}('stick',this)">Stick</button>
          <button class="vbtn" onclick="setStyleA_{pid}('sphere',this)">CPK</button>
          <button class="vbtn" onclick="setStyleA_{pid}('surface',this)">Surface</button>
        </div>
      </div>
      <div class="vp">
        <div class="vp-head">
          <span class="vp-name" style="color:{col_b};">{p['drug_b']}</span>
          <span class="vp-full">{p['fn_b']}</span>
          <span class="vp-cls">{p['cls_b']}</span>
        </div>
        <div id="v3d-b-{pid}" class="mol-viewer"></div>
        <div class="view-btns" id="sbB-{pid}">
          <button class="vbtn active" onclick="setStyleB_{pid}('ballstick',this)">Ball+Stick</button>
          <button class="vbtn" onclick="setStyleB_{pid}('stick',this)">Stick</button>
          <button class="vbtn" onclick="setStyleB_{pid}('sphere',this)">CPK</button>
          <button class="vbtn" onclick="setStyleB_{pid}('surface',this)">Surface</button>
        </div>
      </div>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="showTab_{pid}('props',this)">Molecular Properties</button>
      <button class="tab" onclick="showTab_{pid}('synergy',this)">Synergy by Cell Line</button>
    </div>

    <div id="tab-props-{pid}">
      <div class="props-grid">{pa_html}{pb_html}</div>
    </div>

    <div id="tab-synergy-{pid}" style="display:none;">
      <div class="syn-legend">
        <span style="color:var(--syn);">&#9632; Strong (&gt;10)</span>
        <span style="color:var(--mild);">&#9632; Mild (0&ndash;10)</span>
        <span style="color:var(--neu);">&#9632; Neutral (&minus;10&ndash;0)</span>
        <span style="color:var(--ant);">&#9632; Antagonistic (&lt;&minus;10)</span>
      </div>
      <p class="syn-header">Predicted synergy &mdash; <strong>{p['drug_a']}</strong> + <strong>{p['drug_b']}</strong> &nbsp;&middot;&nbsp; Peak: <strong class="{sc_css}">{max_disp}</strong> &nbsp;in&nbsp; <strong>{top_cl}</strong></p>
      <div class="syn-list" id="hm-{pid}"></div>
    </div>

  </div>
</div>
<script>
(function(){{
  var opened=false, vA=null, vB=null;
  var sdfA=`{sdf_a}`, sdfB=`{sdf_b}`;
  var synData={syn_json};

  function initV(){{
    if(vA) return;
    var elA=document.getElementById('v3d-a-{pid}');
    var elB=document.getElementById('v3d-b-{pid}');
    if(!elA||elA.clientWidth===0||elA.clientHeight===0){{
      setTimeout(initV,50); return;
    }}
    vA=$3Dmol.createViewer(elA,{{backgroundColor:'#0d0f12'}});
    vB=$3Dmol.createViewer(elB,{{backgroundColor:'#0d0f12'}});
    var bs={{stick:{{radius:0.13,colorscheme:'Jmol'}},sphere:{{scale:0.26,colorscheme:'Jmol'}}}};
    [vA,vB].forEach(function(v,i){{
      v.addModel(i===0?sdfA:sdfB,'sdf');
      v.setStyle({{}},bs); v.resize(); v.zoomTo(); v.render(); v.spin('y',0.35);
    }});
    var eA=document.getElementById('v3d-a-{pid}'), eB=document.getElementById('v3d-b-{pid}');
    eA.addEventListener('mouseenter',()=>vA.spin(false));
    eA.addEventListener('mouseleave',()=>vA.spin('y',0.35));
    eB.addEventListener('mouseenter',()=>vB.spin(false));
    eB.addEventListener('mouseleave',()=>vB.spin('y',0.35));
  }}

  function buildHM(){{
    var w=document.getElementById('hm-{pid}');
    if(!w||w.innerHTML!=='') return;
    var yMax=0;
    synData.forEach(r=>{{if(Math.abs(r.val)>yMax) yMax=Math.abs(r.val);}});
    if(!yMax) yMax=1;
    var h='';
    synData.forEach(function(r){{
      var pct=Math.min(Math.abs(r.val)/yMax*100,100);
      var c=r.val>10?'#34d399':r.val>0?'#a3e635':r.val>-10?'#fbbf24':'#f87171';
      h+='<div class="syn-row">'
       + '<span class="syn-cl">'+r.cl+'</span>'
       + '<div class="syn-bar-wrap"><div class="syn-bar" style="width:'+pct+'%;background:'+c+';"></div></div>'
       + '<span class="syn-val" style="color:'+c+';">'+(r.val>=0?'+':'')+r.val+'</span>'
       + '<span class="syn-interp">'+r.interp+'</span>'
       + '</div>';
    }});
    w.innerHTML=h;
  }}

  function applyStyle(v,mode){{
    v.removeAllSurfaces(); v.setStyle({{}},{{}});
    if(mode==='ballstick') v.setStyle({{}},{{stick:{{radius:0.13,colorscheme:'Jmol'}},sphere:{{scale:0.26,colorscheme:'Jmol'}}}});
    else if(mode==='stick')   v.setStyle({{}},{{stick:{{radius:0.12,colorscheme:'Jmol'}}}});
    else if(mode==='sphere')  v.setStyle({{}},{{sphere:{{colorscheme:'Jmol'}}}});
    else if(mode==='surface'){{
      v.setStyle({{}},{{stick:{{radius:0.06,colorscheme:'Jmol'}}}});
      v.addSurface($3Dmol.SurfaceType.VDW,{{opacity:0.7,colorscheme:'Jmol'}});
    }}
    v.render();
  }}

  window['_tp_{pid}']=function(){{
    var body=document.getElementById('body-{pid}');
    var icon=document.getElementById('icon-{pid}');
    var card=document.getElementById('pair-{pid}');
    opened=!opened;
    if(opened){{
      body.style.display='block'; icon.classList.add('open'); card.classList.add('open');
      void body.offsetHeight;
      requestAnimationFrame(function(){{requestAnimationFrame(function(){{initV();}});}});
    }}else{{
      body.style.display='none'; icon.classList.remove('open'); card.classList.remove('open');
    }}
  }};

  window['setStyleA_{pid}']=function(m,btn){{
    document.querySelectorAll('#sbA-{pid} .vbtn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active'); applyStyle(vA,m);
  }};
  window['setStyleB_{pid}']=function(m,btn){{
    document.querySelectorAll('#sbB-{pid} .vbtn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active'); applyStyle(vB,m);
  }};
  window['showTab_{pid}']=function(tab,btn){{
    ['props','synergy'].forEach(t=>{{
      document.getElementById('tab-'+t+'-{pid}').style.display=(t===tab?'block':'none');
    }});
    document.querySelectorAll('#pair-{pid} .tab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    if(tab==='synergy') buildHM();
  }};
}})();
</script>
"""


pairs_html = "\n".join(make_pair_row(p) for p in pair_data)

# Model stats
m         = bundle.get("metrics_coldstart", {})
best_name = bundle.get("model_name","—")
best_r    = m.get(best_name, {}).get("pearson_r", "—")
best_rmse = m.get(best_name, {}).get("rmse", "—")

# Count synergy categories
counts = {"strong":0,"mild":0,"neutral":0,"antagonist":0}
for p in pair_data:
    if p["sc"] in counts: counts[p["sc"]] += 1

FULL_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Drug Synergy Dashboard &mdash; Phase 2</title>
  <meta name="description" content="3D drug synergy predictions &mdash; {len(pair_data)} drug pair combinations across cancer cell lines."/>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:      #111418;
      --surface: #1a1d23;
      --surface2:#21252c;
      --border:  #2a2e38;
      --text:    #e8eaf0;
      --muted:   #7c8494;
      --faint:   #4a5060;
      --accent:  #4f8ef7;
      --syn:     #34d399;
      --ant:     #f87171;
      --mild:    #a3e635;
      --neu:     #fbbf24;
      --radius:  10px;
    }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 15px;
      line-height: 1.5;
      min-height: 100vh;
    }}

    /* ── HEADER ── */
    .page-header {{
      padding: 40px 32px 32px;
      border-bottom: 1px solid var(--border);
      max-width: 1000px;
      margin: 0 auto;
    }}
    .page-header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 6px;
      letter-spacing: -0.3px;
    }}
    .page-header p {{
      font-size: 0.9rem;
      color: var(--muted);
    }}

    /* ── STATS ROW ── */
    .stats-row {{
      max-width: 1000px;
      margin: 24px auto;
      padding: 0 32px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .stat-chip {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 20px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      flex: 1;
      min-width: 120px;
    }}
    .stat-chip .lbl {{ font-size: 0.72rem; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
    .stat-chip .val {{ font-size: 1.25rem; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }}
    .stat-chip.syn .val {{ color: var(--syn); }}
    .stat-chip.ant .val {{ color: var(--ant); }}
    .stat-chip.acc .val {{ color: var(--accent); }}

    /* ── CONTROLS ── */
    .controls {{
      max-width: 1000px;
      margin: 0 auto 16px;
      padding: 0 32px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .search-box {{
      flex: 1;
      min-width: 220px;
      position: relative;
    }}
    .search-box input {{
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-size: 0.88rem;
      font-family: 'Inter', sans-serif;
      padding: 10px 14px 10px 36px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-box input:focus {{ border-color: var(--accent); }}
    .search-box input::placeholder {{ color: var(--faint); }}
    .search-box::before {{
      content: '\2315';
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--muted);
      font-size: 1rem;
      pointer-events: none;
    }}
    .filters {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .fbtn {{
      padding: 8px 14px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
      font-family: 'Inter', sans-serif;
      white-space: nowrap;
    }}
    .fbtn:hover {{ color: var(--text); border-color: #3a3e4a; }}
    .fbtn.active {{ color: #fff; background: var(--accent); border-color: var(--accent); }}
    .fbtn.fs.active  {{ background: transparent; border-color: var(--syn); color: var(--syn); }}
    .fbtn.fm.active  {{ background: transparent; border-color: var(--mild); color: var(--mild); }}
    .fbtn.fn.active  {{ background: transparent; border-color: var(--neu); color: var(--neu); }}
    .fbtn.fa.active  {{ background: transparent; border-color: var(--ant); color: var(--ant); }}
    .sort-sel {{
      padding: 8px 12px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-size: 0.82rem;
      outline: none;
      cursor: pointer;
      font-family: 'Inter', sans-serif;
    }}
    .pair-count {{
      font-size: 0.8rem;
      color: var(--muted);
      white-space: nowrap;
      margin-left: auto;
    }}

    /* ── PAIR LIST ── */
    .pair-list {{
      max-width: 1000px;
      margin: 0 auto;
      padding: 0 32px 60px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    /* ── PAIR CARD ── */
    .pc {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .pc:hover {{ border-color: #3a3e4a; }}
    .pc.open {{ border-color: var(--accent); }}

    /* ── CARD ROW (collapsed) ── */
    .pc-row {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      align-items: center;
      gap: 16px;
      padding: 14px 18px;
      cursor: pointer;
      user-select: none;
    }}
    .pc-row:hover {{ background: rgba(255,255,255,0.02); }}

    .pc-drugs {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .drug-tag {{
      font-size: 0.82rem;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 6px;
      border: 1px solid var(--dc, #4f8ef7);
      color: var(--dc, #4f8ef7);
      background: rgba(79, 142, 247, 0.08);
      letter-spacing: 0.2px;
      white-space: nowrap;
    }}
    .plus-sep {{ color: var(--faint); font-size: 1rem; font-weight: 300; }}
    .pc-sub {{
      font-size: 0.77rem;
      color: var(--muted);
      margin-top: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .pc-score {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 2px;
      flex-shrink: 0;
    }}
    .score-num {{
      font-size: 1rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.5px;
    }}
    .score-lbl {{ font-size: 0.68rem; color: var(--muted); }}
    .sc-s {{ color: var(--syn); }}
    .sc-m {{ color: var(--mild); }}
    .sc-n {{ color: var(--neu); }}
    .sc-a {{ color: var(--ant); }}

    .sc-pill {{
      font-size: 0.67rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .pill-s {{ background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.3); color: var(--syn); }}
    .pill-m {{ background: rgba(163,230,53,.12);  border: 1px solid rgba(163,230,53,.3); color: var(--mild); }}
    .pill-n {{ background: rgba(251,191,36,.12);  border: 1px solid rgba(251,191,36,.3); color: var(--neu); }}
    .pill-a {{ background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.3); color: var(--ant); }}

    .caret {{
      color: var(--muted);
      transition: transform 0.25s;
      flex-shrink: 0;
    }}
    .caret.open {{ transform: rotate(180deg); color: var(--accent); }}

    /* ── EXPANDED BODY ── */
    .pc-body {{
      display: none;
      border-top: 1px solid var(--border);
      padding: 20px 18px 18px;
    }}

    /* 3D viewers */
    .viewers-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 20px;
    }}
    @media(max-width:600px) {{ .viewers-grid {{ grid-template-columns: 1fr; }} }}
    .vp {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .vp-head {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .vp-name {{ font-size: 0.85rem; font-weight: 600; }}
    .vp-full {{ font-size: 0.75rem; color: var(--muted); flex: 1; }}
    .vp-cls  {{ font-size: 0.65rem; color: var(--faint); background: var(--bg); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; white-space: nowrap; }}
    .mol-viewer {{ width: 100%; height: 280px; background: #0d0f12; position: relative; overflow: hidden; }}
    .view-btns {{
      display: flex;
      gap: 4px;
      padding: 8px 10px;
      background: rgba(0,0,0,0.3);
    }}
    .vbtn {{
      font-size: 0.68rem;
      font-family: 'Inter', sans-serif;
      padding: 4px 10px;
      border-radius: 5px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      transition: all 0.15s;
    }}
    .vbtn:hover {{ color: var(--text); border-color: #3a3e4a; }}
    .vbtn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

    /* Tabs */
    .tabs {{
      display: flex;
      border-bottom: 1px solid var(--border);
      margin-bottom: 16px;
    }}
    .tab {{
      padding: 10px 16px;
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--muted);
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: all 0.15s;
      font-family: 'Inter', sans-serif;
    }}
    .tab:hover {{ color: var(--text); }}
    .tab.active {{ color: var(--text); border-bottom-color: var(--accent); }}

    /* Molecular properties */
    .props-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media(max-width:600px) {{ .props-grid {{ grid-template-columns: 1fr; }} }}
    .prop-card {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .prop-card-name {{ font-size: 0.82rem; font-weight: 600; margin-bottom: 10px; }}
    .prop-rows {{ display: flex; flex-direction: column; gap: 6px; }}
    .prop-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
    }}
    .prop-lbl {{ color: var(--muted); }}
    .prop-val {{ font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }}
    .prop-divider {{ border: none; border-top: 1px solid var(--border); margin: 6px 0; }}

    /* Composition bar */
    .comp-bar {{ height: 6px; border-radius: 3px; overflow: hidden; display: flex; margin: 8px 0 4px; }}
    .comp-legend {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }}
    .comp-tag {{ font-size: 0.62rem; padding: 2px 6px; border-radius: 4px; border: 1px solid; display: flex; align-items: center; gap: 3px; }}
    .comp-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}

    /* Synergy chart */
    .syn-header {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }}
    .syn-legend {{ display: flex; gap: 16px; font-size: 0.72rem; color: var(--muted); margin-bottom: 14px; flex-wrap: wrap; }}
    .syn-list {{
      max-height: 360px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .syn-row {{
      display: grid;
      grid-template-columns: 140px 1fr 68px 130px;
      align-items: center;
      gap: 10px;
      padding: 4px 0;
      font-size: 0.78rem;
    }}
    .syn-cl {{ color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .syn-bar-wrap {{ height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }}
    .syn-bar {{ height: 100%; border-radius: 3px; }}
    .syn-val {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; text-align: right; font-size: 0.76rem; }}
    .syn-interp {{ color: var(--muted); font-size: 0.68rem; }}

    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

    .no-data {{ color: var(--muted); font-size: 0.82rem; padding: 16px 0; }}

    @media(max-width:700px) {{
      .pc-row {{ grid-template-columns: 1fr auto; }}
      .sc-pill {{ display: none; }}
      .syn-row {{ grid-template-columns: 110px 1fr 58px; }}
      .syn-interp {{ display: none; }}
      .page-header, .stats-row, .controls, .pair-list {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>

<div class="page-header">
  <h1>Drug Synergy Dashboard</h1>
  <p>Phase 2 &mdash; 3D conformer features &middot; {len(pair_data)} drug pairs &middot; {len(cell_embeddings)} cell lines &middot; {best_name} model</p>
</div>

<div class="stats-row">
  <div class="stat-chip acc">
    <span class="lbl">Drug Pairs</span>
    <span class="val">{len(pair_data)}</span>
  </div>
  <div class="stat-chip">
    <span class="lbl">Cell Lines</span>
    <span class="val">{len(cell_embeddings)}</span>
  </div>
  <div class="stat-chip syn">
    <span class="lbl">Strong Synergy</span>
    <span class="val">{counts['strong']}</span>
  </div>
  <div class="stat-chip">
    <span class="lbl">Mild Synergy</span>
    <span class="val">{counts['mild']}</span>
  </div>
  <div class="stat-chip ant">
    <span class="lbl">Antagonistic</span>
    <span class="val">{counts['antagonist']}</span>
  </div>
</div>

<div class="controls">
  <div class="search-box">
    <input id="searchBox" type="text" placeholder="Search drug name or class&hellip;" oninput="filterPairs()"/>
  </div>
  <div class="filters">
    <button class="fbtn active" id="fb-all" onclick="setFilter('all',this)">All</button>
    <button class="fbtn fs"  onclick="setFilter('strong',this)">Strong</button>
    <button class="fbtn fm"  onclick="setFilter('mild',this)">Mild</button>
    <button class="fbtn fn"  onclick="setFilter('neutral',this)">Neutral</button>
    <button class="fbtn fa"  onclick="setFilter('antagonist',this)">Antagonistic</button>
  </div>
  <select class="sort-sel" id="sortSel" onchange="sortPairs()">
    <option value="default">Default order</option>
    <option value="desc">Highest synergy first</option>
    <option value="asc">Lowest synergy first</option>
    <option value="alpha">A &rarr; Z</option>
  </select>
  <span class="pair-count" id="pairCount"></span>
</div>

<div class="pair-list" id="pairList">
{pairs_html}
</div>

<script>
var curFilter='all';
function togglePair(pid){{ var fn=window['_tp_'+pid]; if(fn) fn(); }}
function filterPairs(){{
  var q=document.getElementById('searchBox').value.toLowerCase();
  var cards=document.querySelectorAll('.pc');
  var v=0;
  cards.forEach(function(c){{
    var da=(c.dataset.da||'').toLowerCase(), db=(c.dataset.db||'').toLowerCase();
    var fa=(c.dataset.fa||'').toLowerCase(), fb=(c.dataset.fb||'').toLowerCase();
    var ca=(c.dataset.ca||'').toLowerCase(), cb=(c.dataset.cb||'').toLowerCase();
    var sc=c.dataset.sc||'';
    var tm=!q||da.includes(q)||db.includes(q)||fa.includes(q)||fb.includes(q)||ca.includes(q)||cb.includes(q);
    var fm=curFilter==='all'||sc===curFilter;
    c.style.display=(tm&&fm)?'':'none';
    if(tm&&fm) v++;
  }});
  document.getElementById('pairCount').textContent=v+' pair'+(v===1?'':'s');
}}
function setFilter(sc,btn){{
  curFilter=sc;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  filterPairs();
}}
function sortPairs(){{
  var mode=document.getElementById('sortSel').value;
  var list=document.getElementById('pairList');
  var cards=Array.from(list.querySelectorAll('.pc'));
  cards.sort(function(a,b){{
    if(mode==='desc'){{ return (parseFloat(b.dataset.avg)||0)-(parseFloat(a.dataset.avg)||0); }}
    if(mode==='asc') {{ return (parseFloat(a.dataset.avg)||0)-(parseFloat(b.dataset.avg)||0); }}
    if(mode==='alpha') return (a.dataset.da||'').localeCompare(b.dataset.da||'');
    return 0;
  }});
  cards.forEach(c=>list.appendChild(c));
  filterPairs();
}}
window.addEventListener('DOMContentLoaded',filterPairs);
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w") as f:
    f.write(FULL_HTML)

print(f"\nDashboard saved -> {OUTPUT_HTML}")
print("Opening in browser...")
webbrowser.open(f"file://{OUTPUT_HTML}")
