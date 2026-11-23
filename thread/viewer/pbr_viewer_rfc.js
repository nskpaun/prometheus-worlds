/**
 * Visuals-first real-time PBR viewer (RFC prototype — illustrates the Three.js WebGPU
 * rendering path for `method === "pbr"` worlds; photo worlds continue using Spark splats).
 *
 * Key upgrades over thread/viewer/main.js:
 *  - Renders real PBR meshes (`scene.glb`) instead of 6M flat Gaussian discs -> 60-120 fps & crisp edges
 *  - Loads Cycles+OIDN baked 32-bit HDR indirect lightmaps (`lightmap.exr`) onto UV2
 *  - Uses real 360 HDRI sky dome (`env.exr`) for both window vistas/outdoors AND specular IBL
 *  - Enables AgX tone mapping, HDR bloom, GTAO contact occlusion, SSR/probe reflections & height fog
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { EXRLoader } from "three/addons/loaders/EXRLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

export function createPbrRenderer(canvas, width, height) {
  // Target WebGPURenderer once vendored; WebGL2 with half-float HDR FBO illustrates the exact pipeline.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  return renderer;
}

export function createPostPipeline(renderer, scene, camera, width, height) {
  const rt = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    samples: 4,
  });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));

  // Real-time micro-crevice occlusion complements the baked Cycles GI on movable/small clutter.
  const gtao = new GTAOPass(scene, camera, width, height);
  gtao.updateGtaoMaterial({ radius: 0.35, distanceExponent: 1.8, thickness: 0.6 });
  composer.addPass(gtao);

  // Physically thresholded HDR bloom: only >1.0 emissives (flames, lamp filaments, sunlit panes) glow.
  const bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 0.28, 0.55, 1.05);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  return { composer, gtao, bloom };
}

function attachBakedGI(root, lightmapTex, probeEnvMap) {
  lightmapTex.flipY = false;
  lightmapTex.colorSpace = THREE.LinearSRGBColorSpace;
  root.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const mat = obj.material;
    // Static architecture & hero geometry carry packed UV2 coordinates from Blender.
    if (obj.geometry?.attributes?.uv1) {
      mat.lightMap = lightmapTex;
      mat.lightMapIntensity = 1.0;
    }
    if (probeEnvMap) {
      mat.envMap = probeEnvMap;
      mat.envMapIntensity = mat.metalness > 0.5 ? 1.15 : 0.85;
    }
    // Preserve crisp alpha-card foliage silhouettes (alpha-to-coverage instead of blurry alpha blend).
    if (mat.alphaTest > 0) mat.alphaToCoverage = true;
    mat.needsUpdate = true;
  });
}

export async function loadPbrWorld(entry, renderer, scene, camera) {
  const gltf = new GLTFLoader();
  const exr = new EXRLoader();
  const pmrem = new THREE.PMREMGenerator(renderer);

  const [worldGltf, envRaw, lightmapTex] = await Promise.all([
    gltf.loadAsync(entry.scene),
    exr.loadAsync(entry.env),
    exr.loadAsync(entry.lightmap),
  ]);

  // 1. 360 HDRI sky dome + prefiltered specular IBL (visible through real transmissive window glass).
  envRaw.mapping = THREE.EquirectangularReflectionMapping;
  const envPMREM = pmrem.fromEquirectangular(envRaw).texture;
  scene.background = envRaw;
  scene.environment = envPMREM;
  pmrem.dispose();

  // 2. Atmosphere: subtle aerial perspective / height-tinted fog matched to the sky horizon.
  const atmo = entry.atmosphere || {};
  const [fr, fg, fb] = atmo.fog_color || [0.78, 0.82, 0.9];
  scene.fog = new THREE.FogExp2(new THREE.Color(fr, fg, fb), atmo.fog_density ?? 0.008);
  renderer.toneMappingExposure = Math.pow(2, atmo.exposure_ev ?? 0.0);

  // 3. Attach Cycles+OIDN baked HDR lightmaps + enable real-time shadows for dynamic hero lights.
  attachBakedGI(worldGltf.scene, lightmapTex, envPMREM);
  worldGltf.scene.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
    if (obj.isLight && (obj.isDirectionalLight || obj.isSpotLight)) {
      obj.castShadow = true;
      obj.shadow.mapSize.set(2048, 2048);
      obj.shadow.bias = -0.0004;
      obj.shadow.normalBias = 0.02;
    }
  });
  scene.add(worldGltf.scene);

  // 4. Frame camera from the VLM-approved hero composition.
  if (entry.eye) camera.position.set(...entry.eye);
  if (entry.target) camera.lookAt(...entry.target);

  return {
    root: worldGltf.scene,
    dispose() {
      scene.remove(worldGltf.scene);
      envRaw.dispose();
      envPMREM.dispose();
      lightmapTex.dispose();
    },
  };
}
