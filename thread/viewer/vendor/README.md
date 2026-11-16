# Vendored libraries

The viewer loads these files directly, so the project needs no Node.js or npm. They are unmodified copies of the published packages.

| Folder | Package | Version | Licence |
|---|---|---|---|
| `three/` | [three](https://github.com/mrdoob/three.js) (`build/three.module.js`, `build/three.core.js`, and the addons the viewer and Spark import) | 0.187.0 | MIT |
| `spark/` | [@sparkjsdev/spark](https://github.com/sparkjsdev/spark) (`dist/spark.module.js`) | 2.4.0 | MIT |
| `rapier/` | [@dimforge/rapier3d-compat](https://github.com/dimforge/rapier.js) (`dist/rapier.mjs`, WebAssembly inlined) | 0.21.0 | Apache-2.0 |

To update one, download the new version's files from the package (for example from the npm registry's tarball) and replace them here. If an addon starts importing another file, copy that too: the browser console names any file it cannot find.
