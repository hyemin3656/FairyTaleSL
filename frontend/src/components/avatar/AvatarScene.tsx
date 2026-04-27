import { useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  VRM,
  VRMLoaderPlugin,
  VRMUtils,
  VRMHumanBoneName,
  VRMExpressionPresetName,
} from "@pixiv/three-vrm";
import type { MotionClip } from "../../api/gloss";

// ── keypoint 파싱 ─────────────────────────────────────────────
function parseKeypoints(kp: number[]) {
  const v3 = (b: number) => new THREE.Vector3(kp[b], kp[b + 1], kp[b + 2]);
  return {
    lhand: Array.from({ length: 21 }, (_, i) => v3(i * 3)),
    rhand: Array.from({ length: 21 }, (_, i) => v3(63 + i * 3)),
  };
}

const isValidHand = (pts: THREE.Vector3[]) =>
  pts.some(p => Math.abs(p.x - 0.5) + Math.abs(p.y - 0.5) > 0.05);

// ── 감정 → VRM 표정 매핑 ─────────────────────────────────────
const EMOTION_TO_VRM: Partial<Record<string, VRMExpressionPresetName>> = {
  happy:     VRMExpressionPresetName.Happy,
  sad:       VRMExpressionPresetName.Sad,
  angry:     VRMExpressionPresetName.Angry,
  surprised: VRMExpressionPresetName.Surprised,
  relaxed:   VRMExpressionPresetName.Relaxed,
};

// ── VRM 손가락 bone 매핑 ──────────────────────────────────────
const FINGER_MAP: { side: "Left" | "Right"; vrm: string[]; mpStart: number }[] = [];

const FINGER_NAMES = [
  { name: "Thumb",  start: 1  },
  { name: "Index",  start: 5  },
  { name: "Middle", start: 9  },
  { name: "Ring",   start: 13 },
  { name: "Little", start: 17 },
];
const JOINT_SUFFIX = ["Proximal", "Intermediate", "Distal"];

for (const side of ["Left", "Right"] as const) {
  for (const { name, start } of FINGER_NAMES) {
    FINGER_MAP.push({
      side,
      vrm: JOINT_SUFFIX.map(s => `${side.toLowerCase()}${name}${s}`),
      mpStart: start,
    });
  }
}

// ── VRM 아바타 컴포넌트 ───────────────────────────────────────
interface VRMAvatarProps {
  clip: MotionClip | null;
  playing: boolean;
}

function VRMAvatar({ clip, playing }: VRMAvatarProps) {
  const [vrm, setVrm] = useState<VRM | null>(null);

  useEffect(() => {
    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    loader.load("/avatar.glb", gltf => {
      const v = gltf.userData.vrm as VRM | undefined;
      if (!v) return;
      VRMUtils.removeUnnecessaryVertices(v.scene);
      VRMUtils.combineSkeletons(v.scene);
      v.scene.traverse(obj => { obj.frustumCulled = false; });
      setVrm(v);
    });
  }, []);

  useFrame((_, delta) => {
    if (!vrm) return;

    const hum = vrm.humanoid;
    const expr = vrm.expressionManager;
    const sp = playing ? 0.25 : 0.08;

    // ── 표정 ─────────────────────────────────────────────────
    const allPresets = [
      VRMExpressionPresetName.Happy,
      VRMExpressionPresetName.Sad,
      VRMExpressionPresetName.Angry,
      VRMExpressionPresetName.Surprised,
      VRMExpressionPresetName.Relaxed,
    ];
    if (expr) {
      const target = clip ? EMOTION_TO_VRM[clip.emotion_label] : undefined;
      for (const p of allPresets) {
        const current = expr.getValue(p) ?? 0;
        const goal = p === target ? 0.85 : 0;
        expr.setValue(p, THREE.MathUtils.lerp(current, goal, sp));
      }
    }

    const lArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftUpperArm);
    const rArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);
    const lFore = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftLowerArm);
    const rFore = hum.getNormalizedBoneNode(VRMHumanBoneName.RightLowerArm);

    // ── 대기 자세: 수어 준비 자세 ─────────────────────────────
    if (!playing || !clip?.keypoints || clip.keypoints.length !== 225) {
      const idle = 0.06;
      // 대기: 차렷 자세 (z 양수=왼팔DOWN, z 음수=오른팔DOWN)
      if (lArm) {
        lArm.rotation.x = THREE.MathUtils.lerp(lArm.rotation.x, 0.1, idle);
        lArm.rotation.y = THREE.MathUtils.lerp(lArm.rotation.y, 0,   idle);
        lArm.rotation.z = THREE.MathUtils.lerp(lArm.rotation.z, 1.5, idle);
      }
      if (rArm) {
        rArm.rotation.x = THREE.MathUtils.lerp(rArm.rotation.x, 0.1, idle);
        rArm.rotation.y = THREE.MathUtils.lerp(rArm.rotation.y, 0,   idle);
        rArm.rotation.z = THREE.MathUtils.lerp(rArm.rotation.z, -1.5, idle);
      }
      if (lFore) { lFore.rotation.x = THREE.MathUtils.lerp(lFore.rotation.x, 0.1, idle); lFore.rotation.y = 0; lFore.rotation.z = 0; }
      if (rFore) { rFore.rotation.x = THREE.MathUtils.lerp(rFore.rotation.x, 0.1, idle); rFore.rotation.y = 0; rFore.rotation.z = 0; }
    } else {
      const { lhand, rhand } = parseKeypoints(clip.keypoints);

      // ── 팔: Euler 회전으로 구동 (부모 본 좌표계 문제 회피) ──
      // wrist.y: 0=화면 위(손 높음), 1=화면 아래(손 낮음)
      // liftFactor: 0(낮음) ~ 0.8(높음)
      const driveArm = (
        armName: VRMHumanBoneName,
        foreName: VRMHumanBoneName,
        wrist: THREE.Vector3,
        isLeft: boolean,
      ) => {
        const armBone = hum.getNormalizedBoneNode(armName);
        const foreBone = hum.getNormalizedBoneNode(foreName);
        if (!armBone) return;

        const signZ = isLeft ? 1 : -1;
        const lift = THREE.MathUtils.clamp((0.65 - wrist.y) * 1.6, 0, 0.8);

        // z: 1.5=차렷, 0.6=수어 대기, lift로 더 올림
        armBone.rotation.x = THREE.MathUtils.lerp(armBone.rotation.x, -0.4 - lift * 0.2, sp);
        armBone.rotation.y = THREE.MathUtils.lerp(armBone.rotation.y, isLeft ? 0.15 : -0.15, sp);
        armBone.rotation.z = THREE.MathUtils.lerp(armBone.rotation.z, signZ * (1.0 - lift * 0.5), sp);

        if (foreBone) {
          const bend = THREE.MathUtils.clamp(0.7 + lift * 0.4, 0.5, 1.2);
          foreBone.rotation.x = THREE.MathUtils.lerp(foreBone.rotation.x, -bend, sp);
          foreBone.rotation.y = THREE.MathUtils.lerp(foreBone.rotation.y, 0, sp);
          foreBone.rotation.z = THREE.MathUtils.lerp(foreBone.rotation.z, 0, sp);
        }
      };

      if (isValidHand(lhand)) driveArm(VRMHumanBoneName.LeftUpperArm,  VRMHumanBoneName.LeftLowerArm,  lhand[0], true);
      if (isValidHand(rhand)) driveArm(VRMHumanBoneName.RightUpperArm, VRMHumanBoneName.RightLowerArm, rhand[0], false);

      // ── 손가락 ──────────────────────────────────────────────
      const applyFingers = (hand: THREE.Vector3[], side: "Left" | "Right") => {
        const isLeft = side === "Left";
        const fingers = FINGER_MAP.filter(f => f.side === side);
        for (const { vrm: boneNames, mpStart } of fingers) {
          for (let j = 0; j < 3; j++) {
            const boneName = boneNames[j] as VRMHumanBoneName;
            const bone = hum.getNormalizedBoneNode(boneName);
            if (!bone) continue;
            const a = hand[mpStart + j];
            const b = hand[mpStart + j + 1];
            if (!a || !b) continue;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const segLen = Math.sqrt(dx * dx + dy * dy) || 0.001;
            const curl = THREE.MathUtils.clamp(dy / segLen * 1.2, -0.2, 1.4);
            const signZ = isLeft ? 1 : -1;
            bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, curl * signZ, sp);
          }
        }
      };

      if (isValidHand(lhand)) applyFingers(lhand, "Left");
      if (isValidHand(rhand)) applyFingers(rhand, "Right");
    }

    // bone 조작 완료 후 VRM 업데이트 (항상 마지막에 호출)
    vrm.update(delta);
  });

  if (!vrm) return null;
  return (
    <primitive
      object={vrm.scene}
      position={[0, -1.2, 0]}
      rotation={[0, Math.PI, 0]}
    />
  );
}

// ── 글로스 오버레이 ───────────────────────────────────────────
interface GlossOverlayProps {
  clip: MotionClip | null;
  status: string;
  currentIndex: number;
  total: number;
}

function GlossOverlay({ clip, status, currentIndex, total }: GlossOverlayProps) {
  return (
    <div className="gloss-overlay">
      {status === "connecting" && (
        <span className="overlay-tag connecting">연결 중…</span>
      )}
      {status === "streaming" && clip && (
        <>
          <span className={`overlay-tag ${clip.is_fallback ? "fallback" : "matched"}`}>
            {clip.gloss}
          </span>
          <span className="overlay-progress">
            {currentIndex + 1} / {total}
          </span>
        </>
      )}
      {status === "done" && (
        <span className="overlay-tag done">완료</span>
      )}
    </div>
  );
}

// ── 메인 AvatarScene ─────────────────────────────────────────
interface AvatarSceneProps {
  clip: MotionClip | null;
  status: string;
  currentIndex: number;
  total: number;
  tokens: string[];
}

export default function AvatarScene({
  clip,
  status,
  currentIndex,
  total,
  tokens,
}: AvatarSceneProps) {
  const playing = status === "streaming";

  return (
    <div className="avatar-scene-wrap">
      <Canvas
        camera={{ position: [0, 0.3, 1.5], fov: 46 }}
        shadows
        style={{ background: "#0f0e1a", borderRadius: 12 }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[1, 3, 2]} intensity={1.4} castShadow />
        <pointLight position={[-1, 2, 2]} intensity={0.5} color="#c4b5fd" />

        <VRMAvatar clip={clip} playing={playing} />
        <Environment preset="city" />

        <OrbitControls
          enablePan={false}
          minDistance={1.2}
          maxDistance={3}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI / 1.8}
          target={[0, 0.25, 0]}
        />
      </Canvas>

      <GlossOverlay
        clip={clip}
        status={status}
        currentIndex={currentIndex}
        total={total}
      />

      {tokens.length > 0 && (
        <div className="gloss-tokens">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`token ${i === currentIndex && playing ? "active" : ""}`}
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
