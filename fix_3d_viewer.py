#!/usr/bin/env python3
"""
Fix 3D molecular viewer centering issues in synergy_dashboard.html.

Problems fixed:
1. viewer-panel has overflow:hidden which can clip WebGL canvas
2. initV() calls zoomTo() before resize() - molecule off-center
3. Toggle uses double-RAF which isn't reliable; need setTimeout with delay
"""

import re

with open('synergy_dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

original_len = len(content)
print(f"File loaded: {original_len} bytes")

# Fix 1: Remove overflow:hidden from .viewer-panel CSS
# This was clipping the WebGL canvas
old_css = '.viewer-panel { background:var(--surface2); border:1px solid var(--border); border-radius:14px; overflow:hidden; }'
new_css = '.viewer-panel { background:var(--surface2); border:1px solid var(--border); border-radius:14px; overflow:visible; }'
count1 = content.count(old_css)
content = content.replace(old_css, new_css)
print(f"Fix 1 (overflow:hidden -> overflow:visible): {count1} replacements")

# Fix 2: Fix initV() - call resize() before zoomTo() so dimensions are correct
# Pattern: v.setStyle({},bs); v.zoomTo(); v.render(); v.spin('y',0.4);
old_init = "v.setStyle({},bs); v.zoomTo(); v.render(); v.spin('y',0.4);"
new_init = "v.setStyle({},bs); v.resize(); v.zoomTo(); v.render(); v.spin('y',0.4);"
count2 = content.count(old_init)
content = content.replace(old_init, new_init)
print(f"Fix 2 (resize before zoomTo in initV): {count2} replacements")

# Fix 3: Replace double-RAF with setTimeout(150ms) for reliable layout completion
# Old: requestAnimationFrame(function(){requestAnimationFrame(function(){initV();if(vA){vA.resize();vA.render();}if(vB){vB.resize();vB.render();}});});
# New: setTimeout(function(){initV();if(vA){vA.resize();vA.zoomTo();vA.render();}if(vB){vB.resize();vB.zoomTo();vB.render();}},150);
old_raf = "requestAnimationFrame(function(){requestAnimationFrame(function(){initV();if(vA){vA.resize();vA.render();}if(vB){vB.resize();vB.render();}});});"
new_raf = "setTimeout(function(){initV();if(vA){vA.resize();vA.zoomTo();vA.render();}if(vB){vB.resize();vB.zoomTo();vB.render();}},150);"
count3 = content.count(old_raf)
content = content.replace(old_raf, new_raf)
print(f"Fix 3 (double-RAF -> setTimeout(150ms) + zoomTo): {count3} replacements")

# Fix 4: Also add mol-viewer canvas position:relative to ensure correct canvas placement
old_mol = '.mol-viewer { width:100%; height:300px; background:#06060f; }'
new_mol = '.mol-viewer { width:100%; height:300px; background:#06060f; position:relative; }'
count4 = content.count(old_mol)
content = content.replace(old_mol, new_mol)
print(f"Fix 4 (mol-viewer position:relative): {count4} replacements")

with open('synergy_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)

new_len = len(content)
print(f"\nDone! File saved: {new_len} bytes (delta: {new_len - original_len:+d})")
print(f"Total fixes: {count1 + count2 + count3 + count4}")
