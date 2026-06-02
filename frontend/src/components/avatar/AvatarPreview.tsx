import { useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from "@pixiv/three-vrm";

function PreviewModel({ url }: { url: string }) {
  const [vrm, setVrm] = useState<VRM | null>(null);

  useEffect(() => {
    setVrm(null);
    const loader = new GLTFLoader();
    loader.register((p) => new VRMLoaderPlugin(p));
    loader.load(url, (gltf) => {
      const v = gltf.userData.vrm as VRM | undefined;
      if (!v) return;
      VRMUtils.removeUnnecessaryVertices(v.scene);
      VRMUtils.combineSkeletons(v.scene);
      v.scene.traverse((o) => { o.frustumCulled = false; });

      // head bone 기준으로 Y 오프셋 계산 (키 무관하게 상반신만 표시)
      v.scene.updateWorldMatrix(true, true);
      const headNode = v.humanoid.getRawBoneNode(VRMHumanBoneName.Head);
      if (headNode) {
        const headPos = new THREE.Vector3();
        headNode.getWorldPosition(headPos);
        v.scene.position.y = 0.2 - headPos.y;
      } else {
        const box = new THREE.Box3().setFromObject(v.scene);
        v.scene.position.y = 0.2 - box.max.y;
      }
      // chibi 모델처럼 실제 머리 메시가 head bone보다 큰 경우 카메라 상한(y≈0.5) 보정
      v.scene.updateWorldMatrix(true, true);
      const previewBox = new THREE.Box3().setFromObject(v.scene);
      if (previewBox.max.y > 0.5) {
        v.scene.position.y -= (previewBox.max.y - 0.5);
      }

      setVrm(v);
    });
  }, [url]);

  useFrame((_, delta) => {
    if (!vrm) return;
    const breathe = Math.sin(Date.now() / 1000 * 1.2) * 0.018;
    const hum = vrm.humanoid;
    const lArm = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftUpperArm);
    const rArm = hum.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);
    const chest = hum.getNormalizedBoneNode(VRMHumanBoneName.Chest);
    if (lArm)  { lArm.rotation.z = THREE.MathUtils.lerp(lArm.rotation.z, 1.4, 0.1); lArm.rotation.x = 0.08; }
    if (rArm)  { rArm.rotation.z = THREE.MathUtils.lerp(rArm.rotation.z, -1.4, 0.1); rArm.rotation.x = 0.08; }
    if (chest) chest.rotation.x = breathe;
    vrm.update(delta);
  });

  if (!vrm) return null;
  return <primitive object={vrm.scene} rotation={[0, Math.PI, 0]} />;
}

export default function AvatarPreview({ url }: { url: string }) {
  return (
    <Canvas
      camera={{ position: [0, 0.2, 0.85], fov: 50 }}
      gl={{ alpha: true }}
      onCreated={({ gl, scene }) => {
        gl.setClearColor(0x000000, 0);
        scene.background = null;
      }}
      style={{ width: "100%", height: "100%" }}
    >
      <ambientLight intensity={0.9} />
      <directionalLight position={[1, 3, 2]} intensity={1.4} />
      <PreviewModel url={url} />
    </Canvas>
  );
}
