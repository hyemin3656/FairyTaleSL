import { useEffect, useRef, useState } from "react";
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

// ── 감정 → VRM 표정 매핑 ─────────────────────────────────────
const EMOTION_TO_VRM: Partial<Record<string, VRMExpressionPresetName>> = {
  happy:     VRMExpressionPresetName.Happy,
  sad:       VRMExpressionPresetName.Sad,
  angry:     VRMExpressionPresetName.Angry,
  surprised: VRMExpressionPresetName.Surprised,
  relaxed:   VRMExpressionPresetName.Relaxed,
};

const ALL_PRESETS = [
  VRMExpressionPresetName.Happy,
  VRMExpressionPresetName.Sad,
  VRMExpressionPresetName.Angry,
  VRMExpressionPresetName.Surprised,
  VRMExpressionPresetName.Relaxed,
];

// ── VRM 손가락 bone 매핑 ──────────────────────────────────────
const FINGER_NAMES = [
  { name: "Thumb",  start: 1  },
  { name: "Index",  start: 5  },
  { name: "Middle", start: 9  },
  { name: "Ring",   start: 13 },
  { name: "Little", start: 17 },
];
const JOINT_SUFFIX = ["Proximal", "Intermediate", "Distal"];

type FingerEntry = { side: "Left" | "Right"; vrm: string[]; mpStart: number };
const FINGER_MAP: FingerEntry[] = [];
for (const side of ["Left", "Right"] as const) {
  for (const { name, start } of FINGER_NAMES) {
    FINGER_MAP.push({
      side,
      vrm: JOINT_SUFFIX.map(s => `${side.toLowerCase()}${name}${s}`),
      mpStart: start,
    });
  }
}

// ── 프레임 선형 보간 ──────────────────────────────────────────
function lerpFrames(f0: number[], f1: number[], t: number): number[] {
  return f0.map((v, i) => v + (f1[i] - v) * t);
}

// ── 포즈 랜드마크 기반 팔 구동 ───────────────────────────────
// 포즈: indices 126..224 (33점 × 3)
// 11=왼쪽어깨, 13=왼쪽팔꿈치, 15=왼쪽손목
// 12=오른쪽어깨, 14=오른쪽팔꿈치, 16=오른쪽손목
function drivePoseBones(kp: number[], hum: VRM["humanoid"], sp: number) {
  const p = (i: number) => ({ x: kp[126 + i * 3], y: kp[126 + i * 3 + 1], z: kp[126 + i * 3 + 2] });
  const lSh = p(11), lEl = p(13), lWr = p(15);
  const rSh = p(12), rEl = p(14), rWr = p(16);

  const lArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftUpperArm);
  const rArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);
  const lFore = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftLowerArm);
  const rFore = hum.getNormalizedBoneNode(VRMHumanBoneName.RightLowerArm);

  const hasL = lSh.y > 0 && lWr.y > 0;
  const hasR = rSh.y > 0 && rWr.y > 0;

  if (hasL && lArm) {
    // dy: 손목이 어깨보다 얼마나 낮은지 (+ = 낮음 = 팔 내림)
    const dy = lWr.y - lSh.y;
    // dx: 손목이 어깨보다 얼마나 몸 안쪽인지 (signer 기준 좌어깨에서 우측 = 몸 중앙 방향)
    const dx = lWr.x - lSh.x;
    const targetZ = THREE.MathUtils.clamp(dy * 2.5 + 0.3, 0.0, 1.5);
    // dx 음수 = 손목이 어깨보다 왼쪽(이미지) = signer 팔이 옆으로 → 앞으로 이동 → VRM x 음수
    const targetX = THREE.MathUtils.clamp(dx * 2.0 - 0.2, -1.2, 0.3);
    lArm.rotation.x = THREE.MathUtils.lerp(lArm.rotation.x, targetX, sp);
    lArm.rotation.z = THREE.MathUtils.lerp(lArm.rotation.z, targetZ, sp);
  }
  if (hasR && rArm) {
    const dy = rWr.y - rSh.y;
    const dx = rWr.x - rSh.x;
    const targetZ = THREE.MathUtils.clamp(dy * 2.5 + 0.3, 0.0, 1.5);
    const targetX = THREE.MathUtils.clamp(dx * 2.0 + 0.2, -0.3, 1.2);
    rArm.rotation.x = THREE.MathUtils.lerp(rArm.rotation.x, targetX, sp);
    rArm.rotation.z = THREE.MathUtils.lerp(rArm.rotation.z, -targetZ, sp);
  }

  // 팔꿈치 굽힘: 어깨-손목 직선 중간점보다 팔꿈치가 위에 있으면 굽힘
  if (hasL && lFore && lEl.y > 0) {
    const midY = (lSh.y + lWr.y) / 2;
    const bend = THREE.MathUtils.clamp((midY - lEl.y) * 3.0, 0.0, 1.5);
    lFore.rotation.x = THREE.MathUtils.lerp(lFore.rotation.x, bend, sp);
    lFore.rotation.y = THREE.MathUtils.lerp(lFore.rotation.y, 0, sp);
    lFore.rotation.z = THREE.MathUtils.lerp(lFore.rotation.z, 0, sp);
  }
  if (hasR && rFore && rEl.y > 0) {
    const midY = (rSh.y + rWr.y) / 2;
    const bend = THREE.MathUtils.clamp((midY - rEl.y) * 3.0, 0.0, 1.5);
    rFore.rotation.x = THREE.MathUtils.lerp(rFore.rotation.x, bend, sp);
    rFore.rotation.y = THREE.MathUtils.lerp(rFore.rotation.y, 0, sp);
    rFore.rotation.z = THREE.MathUtils.lerp(rFore.rotation.z, 0, sp);
  }
}

// ── 손가락 구동 ───────────────────────────────────────────────
// lhand: kp[0..62], rhand: kp[63..125]
function driveHandBones(kp: number[], hum: VRM["humanoid"], sp: number) {
  const v3 = (b: number) => new THREE.Vector3(kp[b], kp[b + 1], kp[b + 2]);
  const lhand = Array.from({ length: 21 }, (_, i) => v3(i * 3));
  const rhand = Array.from({ length: 21 }, (_, i) => v3(63 + i * 3));

  const hasL = lhand.some(p => Math.abs(p.x - 0.5) + Math.abs(p.y - 0.5) > 0.05);
  const hasR = rhand.some(p => Math.abs(p.x - 0.5) + Math.abs(p.y - 0.5) > 0.05);

  for (const { side, vrm: boneNames, mpStart } of FINGER_MAP) {
    const hand = side === "Left" ? lhand : rhand;
    const has  = side === "Left" ? hasL  : hasR;
    if (!has) continue;
    const signZ = side === "Left" ? 1 : -1;

    for (let j = 0; j < 3; j++) {
      const bone = hum.getNormalizedBoneNode(boneNames[j] as VRMHumanBoneName);
      if (!bone) continue;
      const a = hand[mpStart + j];
      const b = hand[mpStart + j + 1];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const segLen = Math.sqrt(dx * dx + dy * dy) || 0.001;
      const curl = THREE.MathUtils.clamp(dy / segLen * 1.2, -0.2, 1.4);
      bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, curl * signZ, sp);
    }
  }
}

// ── 대기 자세 ─────────────────────────────────────────────────
function driveIdlePose(hum: VRM["humanoid"], rate: number) {
  const b = (name: VRMHumanBoneName) => hum.getNormalizedBoneNode(name);
  const lArm  = b(VRMHumanBoneName.LeftUpperArm);
  const rArm  = b(VRMHumanBoneName.RightUpperArm);
  const lFore = b(VRMHumanBoneName.LeftLowerArm);
  const rFore = b(VRMHumanBoneName.RightLowerArm);
  const L = (cur: number, tgt: number) => THREE.MathUtils.lerp(cur, tgt, rate);

  if (lArm)  { lArm.rotation.x  = L(lArm.rotation.x,  0.1); lArm.rotation.y  = L(lArm.rotation.y,  0); lArm.rotation.z  = L(lArm.rotation.z,  1.5); }
  if (rArm)  { rArm.rotation.x  = L(rArm.rotation.x,  0.1); rArm.rotation.y  = L(rArm.rotation.y,  0); rArm.rotation.z  = L(rArm.rotation.z, -1.5); }
  if (lFore) { lFore.rotation.x = L(lFore.rotation.x, 0.1); lFore.rotation.y = 0; lFore.rotation.z = 0; }
  if (rFore) { rFore.rotation.x = L(rFore.rotation.x, 0.1); rFore.rotation.y = 0; rFore.rotation.z = 0; }
}

// ── VRM 아바타 컴포넌트 ───────────────────────────────────────
interface VRMAvatarProps {
  clip: MotionClip | null;
  playing: boolean;
}

function VRMAvatar({ clip, playing }: VRMAvatarProps) {
  const [vrm, setVrm] = useState<VRM | null>(null);
  const elapsedRef  = useRef(0);
  const prevGlossRef = useRef<string | null>(null);

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

    const hum  = vrm.humanoid;
    const expr = vrm.expressionManager;
    const sp   = playing ? 0.3 : 0.08;
    const idle = 0.06;

    // ── 표정 ─────────────────────────────────────────────────
    if (expr) {
      const target = clip ? EMOTION_TO_VRM[clip.emotion_label] : undefined;
      for (const p of ALL_PRESETS) {
        const cur  = expr.getValue(p) ?? 0;
        const goal = p === target ? 0.85 : 0;
        expr.setValue(p, THREE.MathUtils.lerp(cur, goal, sp));
      }
    }

    // ── 클립 전환 시 elapsed 리셋 ─────────────────────────────
    const currentGloss = clip?.gloss ?? null;
    if (currentGloss !== prevGlossRef.current) {
      elapsedRef.current = 0;
      prevGlossRef.current = currentGloss;
    }

    // ── 키포인트 시퀀스 재생 ──────────────────────────────────
    const frames = clip?.keypoints;
    const fps    = clip?.fps ?? 15;

    if (playing && frames && frames.length > 0) {
      elapsedRef.current += delta;

      // 프레임 인덱스 계산 (루프)
      const totalFrames = frames.length;
      const framePos    = (elapsedRef.current * fps) % totalFrames;
      const f0 = Math.floor(framePos);
      const f1 = Math.min(f0 + 1, totalFrames - 1);
      const t  = framePos - f0;

      // 보간된 현재 키포인트
      const kp = lerpFrames(frames[f0], frames[f1], t);

      // 포즈 랜드마크로 팔 구동
      drivePoseBones(kp, hum, sp);
      // 손 랜드마크로 손가락 구동
      driveHandBones(kp, hum, sp);
    } else {
      // 대기 자세
      driveIdlePose(hum, playing ? sp : idle);
    }

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
