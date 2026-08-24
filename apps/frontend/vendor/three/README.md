# Three.js runtime

Pinned local browser dependency for the interactive spatial viewer:

- Three.js `0.180.0`
- `three.module.js` and `three.core.js`
- Official `OrbitControls`, `OBJLoader`, and `MTLLoader` addons
- Source: `https://unpkg.com/three@0.180.0/`
- License: MIT, copyright Three.js authors

The files are served locally by the existing PWA. Room geometry is never bundled here;
the viewer retrieves each scan's generated OBJ, MTL, and entity map through the artifact
API at runtime.
