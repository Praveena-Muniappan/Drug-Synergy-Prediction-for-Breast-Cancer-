import os
import json
import webbrowser
import collections
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ── 1. Drug knowledge base ────────────────────────────────────────────────────
DRUG_INFO = {
    "5-FU": {
        "full_name": "5-Fluorouracil",
        "iupac": "5-fluoro-1H-pyrimidine-2,4-dione",
        "drug_class": "Antimetabolite / Pyrimidine analogue",
        "mechanism": "Inhibits thymidylate synthase (TS), blocking DNA synthesis. Also incorporates into RNA, disrupting its function and triggering cell death.",
        "clinical_use": "Colorectal, breast, head & neck, stomach, and pancreatic cancers.",
        "approval": "FDA-approved (1962)",
        "color": "#ff6b6b",
    },
    "ABT-888": {
        "full_name": "Veliparib",
        "iupac": "2-[(2R)-2-methylpyrrolidin-2-yl]-1H-benzimidazole-4-carboxamide",
        "drug_class": "PARP Inhibitor",
        "mechanism": "Inhibits PARP-1 and PARP-2 enzymes, preventing DNA single-strand break repair, causing double-strand breaks and synthetic lethality in BRCA-deficient cancer cells.",
        "clinical_use": "BRCA-mutated ovarian, breast, and lung cancers (combination regimens).",
        "approval": "Investigational / Clinical trials",
        "color": "#ffa94d",
    },
    "AZD1775": {
        "full_name": "Adavosertib",
        "iupac": "2-allyl-1-(6-(2-hydroxypropan-2-yl)pyridin-2-yl)-6-(4-(2-methylpiperazin-1-yl)phenylamino)-1H-pyrazolo[3,4-d]pyrimidin-3(2H)-one",
        "drug_class": "WEE1 Kinase Inhibitor",
        "mechanism": "Selectively inhibits WEE1 kinase, forcing premature mitotic entry in p53-deficient cancer cells, leading to mitotic catastrophe and apoptosis.",
        "clinical_use": "Advanced solid tumours, ovarian cancer, uterine cancer (clinical trials).",
        "approval": "Investigational",
        "color": "#69db7c",
    },
    "BEZ-235": {
        "full_name": "Dactolisib (NVP-BEZ235)",
        "iupac": "2-methyl-2-[4-(3-methyl-2-oxo-8-quinolin-3-yl-2,3-dihydroimidazo[4,5-c]quinolin-1-yl)phenyl]propanenitrile",
        "drug_class": "Dual PI3K / mTOR Inhibitor",
        "mechanism": "Competitively inhibits PI3Ka, PI3Kb, PI3Kg, PI3Kd and mTORC1/mTORC2, blocking the PI3K-AKT-mTOR survival pathway.",
        "clinical_use": "Advanced solid tumours, neuroendocrine tumours (clinical trials).",
        "approval": "Investigational",
        "color": "#74c0fc",
    },
    "BORTEZOMIB": {
        "full_name": "Bortezomib (Velcade)",
        "iupac": "[(1R)-3-methyl-1-[[(2S)-3-phenyl-2-(pyrazine-2-carbonylamino)propanoyl]amino]butyl]boronic acid",
        "drug_class": "Proteasome Inhibitor (boronic acid warhead)",
        "mechanism": "Reversibly inhibits the 26S proteasome chymotrypsin-like activity, disrupting protein homeostasis and triggering the unfolded protein response and apoptosis.",
        "clinical_use": "Multiple myeloma, mantle cell lymphoma.",
        "approval": "FDA-approved (2003)",
        "color": "#da77f2",
    },
    "CARBOPLATIN": {
        "full_name": "Carboplatin (Paraplatin)",
        "iupac": "azane; cyclobutane-1,1-dicarboxylate; platinum(2+)",
        "drug_class": "Platinum-based DNA cross-linking agent",
        "mechanism": "Forms intra- and inter-strand DNA cross-links via its platinum centre, preventing replication and transcription. Reduced nephrotoxicity vs cisplatin.",
        "clinical_use": "Ovarian, lung, head & neck, bladder, and testicular cancers.",
        "approval": "FDA-approved (1989)",
        "color": "#a9e34b",
    },
    "CYCLOPHOSPHAMIDE": {
        "full_name": "Cyclophosphamide (Cytoxan)",
        "iupac": "2-[bis(2-chloroethyl)amino]-2-oxo-1-oxa-3-aza-2-phosphacyclohexane",
        "drug_class": "Nitrogen mustard alkylating agent (prodrug)",
        "mechanism": "Metabolised by CYP2B6/3A4 to phosphoramide mustard, which cross-links DNA strands via N7-guanine alkylation, inhibiting replication. Also immunosuppressive.",
        "clinical_use": "Breast, ovarian, lymphomas, leukaemias; autoimmune diseases.",
        "approval": "FDA-approved (1959)",
        "color": "#ff8787",
    },
    "DASATINIB": {
        "full_name": "Dasatinib (Sprycel)",
        "iupac": "N-(2-chloro-6-methylphenyl)-2-[[6-[4-(2-hydroxyethyl)piperazin-1-yl]-2-methylpyrimidin-4-yl]amino]-1,3-thiazole-5-carboxamide",
        "drug_class": "BCR-ABL / Src-family Tyrosine Kinase Inhibitor (2nd-gen TKI)",
        "mechanism": "Binds to both active and inactive conformations of BCR-ABL. Also inhibits Src, c-Kit, EPHA2, and PDGFR-b kinases with high potency.",
        "clinical_use": "Chronic myelogenous leukaemia (CML), Philadelphia chromosome-positive ALL.",
        "approval": "FDA-approved (2006)",
        "color": "#ffd43b",
    },
    "DEXAMETHASONE": {
        "full_name": "Dexamethasone",
        "iupac": "(8S,9R,10S,11S,13S,14S,16R,17R)-9-fluoro-11,17-dihydroxy-17-(2-hydroxyacetyl)-10,13,16-trimethyl-hexadecahydro-3H-cyclopenta[a]phenanthren-3-one",
        "drug_class": "Synthetic glucocorticoid / Corticosteroid",
        "mechanism": "Binds glucocorticoid receptor causing nuclear translocation and anti-inflammatory gene expression. Suppresses NF-kB and AP-1 pathways. Potent immunosuppressant.",
        "clinical_use": "Multiple myeloma, lymphoma, cerebral oedema, severe COVID-19, allergic/inflammatory conditions.",
        "approval": "FDA-approved (1958)",
        "color": "#63e6be",
    },
    "DINACICLIB": {
        "full_name": "Dinaciclib (SCH 727965)",
        "iupac": "2-(((1-(3-(1-ethyl-2-methyl-1H-imidazo[4,5-h]quinazolin-9-yl)piperidin-1-yl)ethyl)amino)methyl)pyridine 1-oxide",
        "drug_class": "Pan-CDK Inhibitor (CDK1/2/5/9)",
        "mechanism": "Potent pan-CDK inhibitor. Blocks cell-cycle at G1/S and G2/M checkpoints. Suppresses anti-apoptotic MCL-1 via CDK9 inhibition, triggering apoptosis.",
        "clinical_use": "Chronic lymphocytic leukaemia, multiple myeloma (clinical trials).",
        "approval": "Investigational",
        "color": "#91a7ff",
    },
}

ATOM_COLORS = {
    "C":"#909090","H":"#ffffff","N":"#3050f8","O":"#ff0d0d",
    "F":"#90e050","Cl":"#1ff01f","S":"#ffff30","P":"#ff8000",
    "B":"#ffb5b5","Pt":"#d0d0e0",
}

# ── 2. Load dataset ───────────────────────────────────────────────────────────
print("1. Loading local dataset...")
file_path = "/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl"
df = pd.read_pickle(file_path)

unique_drug1 = df[['Drug1_ID','Drug1']].drop_duplicates(subset='Drug1_ID').reset_index(drop=True)
unique_drug2 = (df[['Drug2_ID','Drug2']].drop_duplicates(subset='Drug2_ID').reset_index(drop=True)
                .rename(columns={'Drug2_ID':'Drug1_ID','Drug2':'Drug1'}))
unique_drugs = (pd.concat([unique_drug1,unique_drug2],ignore_index=True)
                .drop_duplicates(subset='Drug1_ID').head(10))

print(f"\n2. Computing properties & building conformers for {len(unique_drugs)} unique drugs...")

# ── 3. Build card data ────────────────────────────────────────────────────────
cards = []
for _, row in unique_drugs.iterrows():
    drug_id = row['Drug1_ID']
    smiles  = row['Drug1']
    if not smiles:
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [!] Could not parse SMILES for {drug_id}")
        continue

    formula   = rdMolDescriptors.CalcMolFormula(mol)
    mw        = round(Descriptors.MolWt(mol),2)
    logp      = round(Descriptors.MolLogP(mol),2)
    tpsa      = round(Descriptors.TPSA(mol),1)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    rings     = rdMolDescriptors.CalcNumRings(mol)
    rotb      = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy_ats = mol.GetNumHeavyAtoms()
    atom_counts = collections.Counter(a.GetSymbol() for a in mol.GetAtoms())
    total_heavy = sum(atom_counts.values())
    atom_comp   = {sym:(cnt,round(cnt/total_heavy*100,1))
                   for sym,cnt in sorted(atom_counts.items(),key=lambda x:-x[1])}

    mol3d = Chem.AddHs(mol)
    res   = AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
    if res != 0:
        print(f"  [!] 3D embedding failed for {drug_id}")
        continue
    AllChem.MMFFOptimizeMolecule(mol3d)
    sdf_block = Chem.MolToMolBlock(mol3d)

    info = DRUG_INFO.get(drug_id,{})
    cards.append({
        "drug_id":    drug_id,
        "smiles":     smiles,
        "sdf_block":  sdf_block,
        "formula":    formula,"mw":mw,"logp":logp,"tpsa":tpsa,
        "hbd":hbd,"hba":hba,"rings":rings,"rotb":rotb,"heavy_ats":heavy_ats,
        "atom_comp":  atom_comp,
        "full_name":  info.get("full_name",  drug_id),
        "iupac":      info.get("iupac",      "—"),
        "drug_class": info.get("drug_class", "—"),
        "mechanism":  info.get("mechanism",  "—"),
        "clinical":   info.get("clinical_use","—"),
        "approval":   info.get("approval",   "—"),
        "color":      info.get("color",      "#6c63ff"),
    })
    print(f"  [{len(cards)}/10] \u2713 {drug_id}  ({formula}, MW={mw})")

# ── 4. HTML helpers ───────────────────────────────────────────────────────────
def atom_bar_html(atom_comp, colors):
    bars   = []
    legend = []
    for sym,(cnt,pct) in atom_comp.items():
        c = colors.get(sym,"#aaa")
        bars.append(f'<div style="width:{pct}%;background:{c};height:100%;display:inline-block;" title="{sym}:{pct}%"></div>')
        legend.append(f'<span class="atom-chip" style="background:{c}20;border-color:{c};">'
                      f'<span class="atom-dot" style="background:{c};"></span>{sym}: {cnt} ({pct}%)</span>')
    return (f'<div class="comp-bar">{"".join(bars)}</div>'
            f'<div class="comp-legend">{"".join(legend)}</div>')

def make_card(c):
    sid      = c["drug_id"].replace(" ","_").replace("-","_").replace("+","plus")
    sdf_safe = c["sdf_block"].replace("`","\\`").replace("${","\\${")
    abar     = atom_bar_html(c["atom_comp"], ATOM_COLORS)
    accent   = c["color"]

    return f"""
<div class="molecule-card" id="card-{sid}">
  <div class="card-header">
    <div class="drug-name-block">
      <span class="drug-badge" style="background:{accent}22;border-color:{accent};color:{accent};">{c['drug_id']}</span>
      <span class="full-name">{c['full_name']}</span>
      <span class="drug-class-tag">{c['drug_class']}</span>
    </div>
    <span class="approval-tag">{c['approval']}</span>
  </div>

  <div class="viewer-wrap">
    <div id="viewer-{sid}" class="mol-viewer"></div>
    <div class="style-btns">
      <button class="style-btn active" onclick="setStyle_{sid}('ballstick',this)">Ball+Stick</button>
      <button class="style-btn" onclick="setStyle_{sid}('stick',this)">Stick</button>
      <button class="style-btn" onclick="setStyle_{sid}('sphere',this)">CPK</button>
      <button class="style-btn" onclick="setStyle_{sid}('surface',this)">Surface</button>
    </div>
  </div>

  <div class="info-tabs">
    <button class="tab-btn active" onclick="showTab_{sid}('props',this)">&#9883; Properties</button>
    <button class="tab-btn" onclick="showTab_{sid}('chem',this)">&#128138; Chemistry</button>
    <button class="tab-btn" onclick="showTab_{sid}('smiles',this)">&#128257; SMILES / 3D</button>
  </div>

  <div id="tab-props-{sid}" class="tab-content">
    <div class="props-grid">
      <div class="prop-item"><span class="prop-label">Formula</span><span class="prop-val mono">{c['formula']}</span></div>
      <div class="prop-item"><span class="prop-label">Mol. Weight</span><span class="prop-val">{c['mw']} g/mol</span></div>
      <div class="prop-item"><span class="prop-label">LogP</span><span class="prop-val">{c['logp']}</span></div>
      <div class="prop-item"><span class="prop-label">TPSA</span><span class="prop-val">{c['tpsa']} &#8491;&#178;</span></div>
      <div class="prop-item"><span class="prop-label">HB Donors</span><span class="prop-val">{c['hbd']}</span></div>
      <div class="prop-item"><span class="prop-label">HB Acceptors</span><span class="prop-val">{c['hba']}</span></div>
      <div class="prop-item"><span class="prop-label">Rings</span><span class="prop-val">{c['rings']}</span></div>
      <div class="prop-item"><span class="prop-label">Rot. Bonds</span><span class="prop-val">{c['rotb']}</span></div>
      <div class="prop-item"><span class="prop-label">Heavy Atoms</span><span class="prop-val">{c['heavy_ats']}</span></div>
    </div>
    <div class="comp-section">
      <div class="comp-title">Atom Composition (heavy atoms)</div>
      {abar}
    </div>
  </div>

  <div id="tab-chem-{sid}" class="tab-content" style="display:none;">
    <table class="chem-table">
      <tr><td class="ct-label">IUPAC Name</td><td class="mono small">{c['iupac']}</td></tr>
      <tr><td class="ct-label">Mechanism</td><td>{c['mechanism']}</td></tr>
      <tr><td class="ct-label">Clinical Use</td><td>{c['clinical']}</td></tr>
    </table>
  </div>

  <div id="tab-smiles-{sid}" class="tab-content" style="display:none;">
    <div class="smiles-section">
      <div class="smiles-label">Canonical SMILES</div>
      <div class="smiles-box">{c['smiles']}</div>
      <div class="repr-header">3D Representation Modes</div>
      <ul class="repr-list">
        <li><strong>Ball + Stick</strong> — Atoms as spheres + bonds as cylinders. Best for reading connectivity &amp; bond angles.</li>
        <li><strong>Stick (Wire)</strong> — Bonds only, no atom spheres. Best for large/complex molecules where overlap hides details.</li>
        <li><strong>CPK (Space-fill)</strong> — Each atom rendered at its Van der Waals radius. Shows true molecular volume and steric bulk.</li>
        <li><strong>Molecular Surface (VDW)</strong> — Solvent-accessible surface coloured by element. Reveals binding pocket shape and electrostatic topology.</li>
      </ul>
    </div>
  </div>
</div>

<script>
(function(){{
  var el     = document.getElementById('viewer-{sid}');
  var viewer = $3Dmol.createViewer(el, {{backgroundColor:'#080818'}});
  var molData = `{sdf_safe}`;
  viewer.addModel(molData,'sdf');
  viewer.setStyle({{}},{{stick:{{radius:0.13,colorscheme:'Jmol'}},sphere:{{scale:0.27,colorscheme:'Jmol'}}}});
  viewer.zoomTo(); viewer.render(); viewer.spin('y',0.5);
  el.addEventListener('mouseenter',function(){{ viewer.spin(false); }});
  el.addEventListener('mouseleave',function(){{ viewer.spin('y',0.5); }});

  window['setStyle_{sid}'] = function(mode,btn){{
    document.querySelectorAll('#card-{sid} .style-btn').forEach(function(b){{b.classList.remove('active');}});
    btn.classList.add('active');
    viewer.removeAllSurfaces();
    viewer.setStyle({{}},{{}});
    if(mode==='ballstick'){{
      viewer.setStyle({{}},{{stick:{{radius:0.13,colorscheme:'Jmol'}},sphere:{{scale:0.27,colorscheme:'Jmol'}}}});
    }}else if(mode==='stick'){{
      viewer.setStyle({{}},{{stick:{{radius:0.12,colorscheme:'Jmol'}}}});
    }}else if(mode==='sphere'){{
      viewer.setStyle({{}},{{sphere:{{colorscheme:'Jmol'}}}});
    }}else if(mode==='surface'){{
      viewer.setStyle({{}},{{stick:{{radius:0.07,colorscheme:'Jmol'}}}});
      viewer.addSurface($3Dmol.SurfaceType.VDW,{{opacity:0.72,colorscheme:'Jmol'}});
    }}
    viewer.render();
  }};

  window['showTab_{sid}'] = function(tab,btn){{
    ['props','chem','smiles'].forEach(function(t){{
      document.getElementById('tab-'+t+'-{sid}').style.display=(t===tab)?'block':'none';
    }});
    document.querySelectorAll('#card-{sid} .tab-btn').forEach(function(b){{b.classList.remove('active');}});
    btn.classList.add('active');
  }};
}})();
</script>
"""

cards_html = "\n".join(make_card(c) for c in cards)

# ── 5. Full HTML ──────────────────────────────────────────────────────────────
FULL_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>3D Drug Conformer Dashboard — Oncology</title>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    :root{{
      --bg:#07070f;--surface:#0f0f22;--surface2:#14143a;
      --border:#1c1c3a;--accent:#6c63ff;--accent2:#00d4ff;
      --text:#dde0f5;--muted:#6060a0;--green:#51cf66;
    }}
    body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:32px 20px 80px;}}

    header{{text-align:center;margin-bottom:48px;}}
    header h1{{font-size:clamp(1.5rem,3.5vw,2.4rem);font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.5px;}}
    header p{{color:var(--muted);font-size:.9rem;margin-top:10px;line-height:1.7;}}

    .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:28px;max-width:1500px;margin:auto;}}

    .molecule-card{{background:var(--surface);border:1px solid var(--border);border-radius:18px;overflow:hidden;transition:transform .25s ease,box-shadow .25s ease;}}
    .molecule-card:hover{{transform:translateY(-5px);box-shadow:0 16px 48px rgba(108,99,255,.22);}}

    .card-header{{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:14px 16px;border-bottom:1px solid var(--border);background:rgba(255,255,255,.025);}}
    .drug-name-block{{display:flex;flex-direction:column;gap:4px;flex:1;}}
    .drug-badge{{display:inline-block;font-weight:700;font-size:.8rem;padding:3px 11px;border-radius:20px;border:1px solid;width:fit-content;letter-spacing:.3px;}}
    .full-name{{font-size:.9rem;font-weight:600;color:var(--text);margin-top:2px;}}
    .drug-class-tag{{font-size:.68rem;color:var(--muted);margin-top:1px;}}
    .approval-tag{{font-size:.65rem;color:var(--green);border:1px solid var(--green);border-radius:20px;padding:2px 8px;white-space:nowrap;margin-top:4px;background:rgba(81,207,102,.1);flex-shrink:0;}}

    .viewer-wrap{{position:relative;}}
    .mol-viewer{{width:100%;height:320px;background:#080818;}}
    .style-btns{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);display:flex;gap:6px;background:rgba(0,0,0,.7);padding:5px 8px;border-radius:20px;backdrop-filter:blur(8px);}}
    .style-btn{{font-size:.67rem;font-family:'Inter',sans-serif;font-weight:500;padding:3px 10px;border-radius:12px;cursor:pointer;border:1px solid rgba(255,255,255,.2);background:transparent;color:#aaa;transition:all .2s;}}
    .style-btn:hover{{background:rgba(255,255,255,.1);color:#fff;}}
    .style-btn.active{{background:var(--accent);border-color:var(--accent);color:#fff;}}

    .info-tabs{{display:flex;border-bottom:1px solid var(--border);}}
    .tab-btn{{flex:1;padding:9px 4px;font-size:.73rem;font-weight:600;background:none;border:none;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;}}
    .tab-btn:hover{{color:var(--text);}}
    .tab-btn.active{{color:var(--accent2);border-bottom-color:var(--accent2);}}
    .tab-content{{padding:14px 16px;}}

    .props-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:12px;}}
    .prop-item{{background:var(--surface2);border-radius:8px;padding:8px 10px;border:1px solid var(--border);}}
    .prop-label{{display:block;font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;}}
    .prop-val{{font-size:.82rem;font-weight:600;color:var(--text);}}
    .mono{{font-family:'JetBrains Mono',monospace;}}

    .comp-section{{margin-top:4px;}}
    .comp-title{{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}}
    .comp-bar{{height:10px;border-radius:6px;overflow:hidden;display:flex;width:100%;margin-bottom:8px;}}
    .comp-legend{{display:flex;flex-wrap:wrap;gap:5px;}}
    .atom-chip{{font-size:.63rem;padding:2px 7px;border-radius:10px;border:1px solid;display:flex;align-items:center;gap:4px;color:var(--text);}}
    .atom-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}

    .chem-table{{width:100%;border-collapse:collapse;font-size:.77rem;}}
    .chem-table tr{{border-bottom:1px solid var(--border);}}
    .chem-table tr:last-child{{border-bottom:none;}}
    .chem-table td{{padding:9px 6px;vertical-align:top;line-height:1.55;}}
    .ct-label{{color:var(--muted);font-weight:600;white-space:nowrap;padding-right:12px;font-size:.68rem;text-transform:uppercase;width:100px;}}
    .small{{font-size:.69rem;}}

    .smiles-section{{font-size:.77rem;}}
    .smiles-label{{color:var(--muted);font-size:.65rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;}}
    .smiles-box{{font-family:'JetBrains Mono',monospace;font-size:.7rem;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px;word-break:break-all;color:#a0d0ff;line-height:1.6;}}
    .repr-header{{color:var(--accent2);font-weight:600;font-size:.75rem;margin:12px 0 6px;}}
    .repr-list{{padding-left:16px;line-height:2.0;color:var(--text);font-size:.75rem;}}
    .repr-list strong{{color:var(--accent2);}}
  </style>
</head>
<body>
<header>
  <h1>&#129516; Oncology Drug 3D Conformer Dashboard</h1>
  <p>
    Topology-Aware Drug Synergy Engine &nbsp;|&nbsp; {len(cards)} unique oncology molecules
    &nbsp;|&nbsp; Interactive WebGL 3D
    &nbsp;&bull;&nbsp; Drag to rotate &nbsp;&bull;&nbsp; Scroll to zoom &nbsp;&bull;&nbsp; Right-click to pan
  </p>
</header>
<div class="grid">
{cards_html}
</div>
</body>
</html>
"""

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_3d_dashboard.html")
with open(output_path, "w") as f:
    f.write(FULL_HTML)

print(f"\n\u2705  Dashboard saved \u2192 {output_path}")
print("\U0001f310  Opening in your default browser...")
webbrowser.open(f"file://{output_path}")
