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
import type { MotionClip, AnimData } from "../../api/gloss";

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
// MediaPipe 좌표: x=이미지 오른쪽(=피사체 왼쪽), y=아래, z=카메라 방향(음수=앞)
// 11=왼쪽어깨, 13=왼쪽팔꿈치, 15=왼쪽손목
// 12=오른쪽어깨, 14=오른쪽팔꿈치, 16=오른쪽손목
function drivePoseBones(kp: number[], hum: VRM["humanoid"], sp: number) {
  type P3 = { x: number; y: number; z: number };
  const p = (i: number): P3 => ({
    x: kp[126 + i * 3], y: kp[126 + i * 3 + 1], z: kp[126 + i * 3 + 2],
  });
  const lSh = p(11), lEl = p(13), lWr = p(15);
  const rSh = p(12), rEl = p(14), rWr = p(16);

  const lArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftUpperArm);
  const rArm  = hum.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);
  const lFore = hum.getNormalizedBoneNode(VRMHumanBoneName.LeftLowerArm);
  const rFore = hum.getNormalizedBoneNode(VRMHumanBoneName.RightLowerArm);

  // 팔꿈치 기준으로 유효성 판단 (손목보다 안정적)
  const hasL = lSh.y > 0 && lEl.y !== 0;
  const hasR = rSh.y > 0 && rEl.y !== 0;

  function sub(a: P3, b: P3): P3 { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; }
  function norm(v: P3): P3 {
    const len = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) || 0.001;
    return { x: v.x/len, y: v.y/len, z: v.z/len };
  }
  function angle(a: P3, b: P3): number {
    const na = norm(a), nb = norm(b);
    return Math.acos(THREE.MathUtils.clamp(na.x*nb.x + na.y*nb.y + na.z*nb.z, -1, 1));
  }

  // ── 상완: 어깨→팔꿈치 방향으로 계산 ─────────────────────────
  // 이전에는 어깨→손목을 사용했으나, 상완 방향은 어깨→팔꿈치가 정확함
  if (hasL && lArm) {
    const d = sub(lEl, lSh);
    // d.y: +0.25≈팔 자연하강, 0≈T포즈, -0.2≈팔 올림 → VRM rz: 1.5=아래, 0=T포즈, -1.5=위
    const targetZ = THREE.MathUtils.clamp(d.y * 6.0, -1.5, 1.5);
    // Y: 팔 전방 회전 — 음수 방향이 몸통 안으로 파고드므로 하한을 -0.4로 제한
    const targetY = THREE.MathUtils.clamp(d.z * 5.0, -0.4, 0.8);
    const targetX = THREE.MathUtils.clamp(d.x * 3.0, -1.0, 0.8);

    lArm.rotation.x = THREE.MathUtils.lerp(lArm.rotation.x, targetX, sp);
    lArm.rotation.y = THREE.MathUtils.lerp(lArm.rotation.y, targetY, sp);
    lArm.rotation.z = THREE.MathUtils.lerp(lArm.rotation.z, targetZ, sp);
  }
  if (hasR && rArm) {
    const d = sub(rEl, rSh);
    const targetZ = THREE.MathUtils.clamp(-d.y * 6.0, -1.5, 1.5);
    // Y: 팔 전방 회전 — 양수 방향이 몸통 안으로 파고드므로 상한을 0.4로 제한
    const targetY = THREE.MathUtils.clamp(-d.z * 5.0, -0.8, 0.4);
    const targetX = THREE.MathUtils.clamp(-d.x * 3.0, -0.8, 1.0);

    rArm.rotation.x = THREE.MathUtils.lerp(rArm.rotation.x, targetX, sp);
    rArm.rotation.y = THREE.MathUtils.lerp(rArm.rotation.y, targetY, sp);
    rArm.rotation.z = THREE.MathUtils.lerp(rArm.rotation.z, targetZ, sp);
  }

  // ── 전완: 어깨→팔꿈치와 팔꿈치→손목의 각도 = 팔꿈치 굽힘 ───
  if (hasL && lFore) {
    // 전완 Y회전도 몸통 관통 방지를 위해 하한 제한
    const bend = THREE.MathUtils.clamp(angle(sub(lEl, lSh), sub(lWr, lEl)), 0, 1.6);
    lFore.rotation.x = THREE.MathUtils.lerp(lFore.rotation.x, 0, sp);
    lFore.rotation.y = THREE.MathUtils.lerp(lFore.rotation.y, -bend, sp);
    lFore.rotation.z = THREE.MathUtils.lerp(lFore.rotation.z, 0, sp);
  }
  if (hasR && rFore) {
    const bend = THREE.MathUtils.clamp(angle(sub(rEl, rSh), sub(rWr, rEl)), 0, 1.6);
    rFore.rotation.x = THREE.MathUtils.lerp(rFore.rotation.x, 0, sp);
    rFore.rotation.y = THREE.MathUtils.lerp(rFore.rotation.y, bend, sp);
    rFore.rotation.z = THREE.MathUtils.lerp(rFore.rotation.z, 0, sp);
  }
}

// ── 손가락 구동 + 손목 회전 ───────────────────────────────────
// lhand: kp[0..62], rhand: kp[63..125]
// MediaPipe 손 랜드마크: x=오른쪽, y=아래, z=카메라방향(음수=가까움)
function driveHandBones(kp: number[], hum: VRM["humanoid"], sp: number) {
  type MP = { x: number; y: number; z: number };
  const mp = (b: number): MP => ({ x: kp[b], y: kp[b + 1], z: kp[b + 2] });
  const lhand = Array.from({ length: 21 }, (_, i) => mp(i * 3));
  const rhand = Array.from({ length: 21 }, (_, i) => mp(63 + i * 3));

  // 손 감지 여부: 원점(0,0)에서 벗어났는지
  const hasL = lhand.some(p => Math.abs(p.x) + Math.abs(p.y) > 0.01);
  const hasR = rhand.some(p => Math.abs(p.x) + Math.abs(p.y) > 0.01);

  function applyHand(hand: MP[], has: boolean, side: "Left" | "Right") {
    const signZ = side === "Left" ? 1 : -1;

    // ── 손목 bone ─────────────────────────────────────────────
    const wristBone = hum.getNormalizedBoneNode(
      side === "Left" ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand
    );

    if (!has) {
      // 손 미감지 시 손목 중립으로 복귀
      if (wristBone) {
        wristBone.rotation.x = THREE.MathUtils.lerp(wristBone.rotation.x, 0, sp);
        wristBone.rotation.y = THREE.MathUtils.lerp(wristBone.rotation.y, 0, sp);
        wristBone.rotation.z = THREE.MathUtils.lerp(wristBone.rotation.z, 0, sp);
      }
      return;
    }

    // ── 손바닥 좌표계 계산 ────────────────────────────────────
    const w = hand[0];   // 손목
    const iM = hand[5];  // 검지 MCP
    const mM = hand[9];  // 중지 MCP (손가락 방향 기준)
    const pM = hand[17]; // 새끼 MCP

    // 손가락 방향 (wrist→middleMCP), 이미지 공간
    const fdx = mM.x - w.x, fdy = mM.y - w.y;
    const fdLen = Math.sqrt(fdx * fdx + fdy * fdy) || 0.001;
    const fdxN = fdx / fdLen, fdyN = fdy / fdLen;

    // 손바닥 가로 방향 (pinkyMCP→indexMCP)
    const pdx = iM.x - pM.x, pdy = iM.y - pM.y;

    // 손바닥 법선 z 성분 (2D cross product, 이미지 평면)
    // V1 = indexMCP - wrist, V2 = pinkyMCP - wrist
    const v1x = iM.x - w.x, v1y = iM.y - w.y;
    const v2x = pM.x - w.x, v2y = pM.y - w.y;
    const normZ = v1x * v2y - v1y * v2x;

    // ── 손목 회전 ─────────────────────────────────────────────
    if (wristBone) {
      // 손가락 방향이 위쪽일 때(fdy<0) vs 수평/아래 → rotation.x 조정
      // atan2(fdx, -fdy): 손가락이 위를 향하면 0, 오른쪽이면 π/2
      const fingerUpAngle = Math.atan2(fdxN, -fdyN);

      // 손바닥 방향 (normZ<0 = 카메라 향함 = supination, >0 = 반대 = pronation)
      // 손바닥 가로 벡터의 y 성분으로 기울기 추정
      const tiltAngle = Math.atan2(pdy, pdx);  // 가로 방향의 기울기

      wristBone.rotation.x = THREE.MathUtils.lerp(
        wristBone.rotation.x,
        THREE.MathUtils.clamp(fingerUpAngle * 0.4 * signZ, -0.8, 0.8),
        sp
      );
      wristBone.rotation.y = THREE.MathUtils.lerp(
        wristBone.rotation.y,
        THREE.MathUtils.clamp(-normZ * 3.0 * signZ, -1.2, 1.2),
        sp
      );
      wristBone.rotation.z = THREE.MathUtils.lerp(
        wristBone.rotation.z,
        THREE.MathUtils.clamp(tiltAngle * 0.5 * signZ, -1.0, 1.0),
        sp
      );
    }

    // ── 손가락 curl (palm-local 공간 기준) ────────────────────
    for (const { side: fSide, vrm: boneNames, mpStart } of FINGER_MAP) {
      if (fSide !== side) continue;
      for (let j = 0; j < 3; j++) {
        const bone = hum.getNormalizedBoneNode(boneNames[j] as VRMHumanBoneName);
        if (!bone) continue;
        const a = hand[mpStart + j];
        const b = hand[mpStart + j + 1];
        if (!a || !b) continue;

        const dx = b.x - a.x, dy = b.y - a.y;
        const segLen = Math.sqrt(dx * dx + dy * dy) || 0.001;

        // 손가락 방향 기준 curl: 1 - (segment · palmForward / segLen)
        // palmForward = (fdxN, fdyN), segment = (dx, dy)
        const fwdDot = (dx * fdxN + dy * fdyN) / segLen;
        // fwdDot≈1 = 쭉 펴짐, fwdDot≈-1 = 완전 굽힘
        const curl = THREE.MathUtils.clamp((1.0 - fwdDot) * 0.8, 0.0, 1.4);

        bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, curl * signZ, sp);
      }
    }
  }

  applyHand(lhand, hasL, "Left");
  applyHand(rhand, hasR, "Right");
}

// ── 사전 계산된 본 회전 데이터로 아바타 구동 ─────────────────
const ANIM_BONE_MAP: Record<string, VRMHumanBoneName> = {
  lArm:  VRMHumanBoneName.LeftUpperArm,
  rArm:  VRMHumanBoneName.RightUpperArm,
  lFore: VRMHumanBoneName.LeftLowerArm,
  rFore: VRMHumanBoneName.RightLowerArm,
  lHand: VRMHumanBoneName.LeftHand,
  rHand: VRMHumanBoneName.RightHand,
};

// {prefix}{Name}{j} → VRMHumanBoneName (손가락 스칼라 curl 매핑)
const FINGER_ANIM_MAP: Record<string, VRMHumanBoneName> = (() => {
  const map: Record<string, VRMHumanBoneName> = {};
  for (const side of ["l", "r"] as const) {
    const sideStr = side === "l" ? "Left" : "Right";
    for (const { name } of FINGER_NAMES) {
      for (let j = 0; j < 3; j++) {
        const key = `${side}${name}${j}`;
        map[key] = `${sideStr.toLowerCase()}${name}${JOINT_SUFFIX[j]}` as VRMHumanBoneName;
      }
    }
  }
  return map;
})();

// 상완 Y 회전 클램프: 몸통 관통 방지 (drivePoseBones와 동일 기준 적용)
const ARM_Y_CLAMP: Record<string, [number, number]> = {
  lArm: [-0.4, 0.8],
  rArm: [-0.8, 0.4],
};

function driveAnimData(anim: AnimData, elapsed: number, hum: VRM["humanoid"], sp: number) {
  const { fps, n, bones } = anim;
  const framePos = (elapsed * fps) % n;
  const f0 = Math.floor(framePos);
  const f1 = Math.min(f0 + 1, n - 1);
  const t  = framePos - f0;

  // 3채널 본 (상완/전완/손목)
  for (const [key, boneName] of Object.entries(ANIM_BONE_MAP)) {
    const frames = bones[key] as [number, number, number][] | undefined;
    if (!frames || frames.length === 0) continue;
    const bone = hum.getNormalizedBoneNode(boneName);
    if (!bone) continue;
    const fr0 = frames[Math.min(f0, frames.length - 1)];
    const fr1 = frames[Math.min(f1, frames.length - 1)];

    const rx = fr0[0] + (fr1[0] - fr0[0]) * t;
    let   ry = fr0[1] + (fr1[1] - fr0[1]) * t;
    const rz = fr0[2] + (fr1[2] - fr0[2]) * t;

    // 상완 Y 클램프: keypoint 노이즈로 인한 팔 몸통 관통 방지
    const yClamp = ARM_Y_CLAMP[key];
    if (yClamp) ry = THREE.MathUtils.clamp(ry, yClamp[0], yClamp[1]);

    bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, rx, sp);
    bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, ry, sp);
    bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, rz, sp);
  }

  // 손가락 curl (스칼라 리스트)
  for (const [key, boneName] of Object.entries(FINGER_ANIM_MAP)) {
    const frames = bones[key] as number[] | undefined;
    if (!frames || frames.length === 0) continue;
    const bone = hum.getNormalizedBoneNode(boneName);
    if (!bone) continue;
    const v0   = frames[Math.min(f0, frames.length - 1)] ?? 0;
    const v1   = frames[Math.min(f1, frames.length - 1)] ?? 0;
    const curl = v0 + (v1 - v0) * t;
    bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, curl, sp);
  }
}

// ── 대기 자세 (미묘한 호흡 포함) ──────────────────────────────
function driveIdlePose(hum: VRM["humanoid"], rate: number) {
  const b = (name: VRMHumanBoneName) => hum.getNormalizedBoneNode(name);
  const lArm  = b(VRMHumanBoneName.LeftUpperArm);
  const rArm  = b(VRMHumanBoneName.RightUpperArm);
  const lFore = b(VRMHumanBoneName.LeftLowerArm);
  const rFore = b(VRMHumanBoneName.RightLowerArm);
  const chest = b(VRMHumanBoneName.Chest);
  const spine = b(VRMHumanBoneName.Spine);
  const L = (cur: number, tgt: number) => THREE.MathUtils.lerp(cur, tgt, rate);

  // 호흡 사이클 (1.2 rad/s ≈ 0.19 Hz)
  const breathe = Math.sin(Date.now() / 1000 * 1.2) * 0.018;

  if (lArm)  { lArm.rotation.x  = L(lArm.rotation.x,  0.1); lArm.rotation.y  = L(lArm.rotation.y,  0); lArm.rotation.z  = L(lArm.rotation.z,  1.5); }
  if (rArm)  { rArm.rotation.x  = L(rArm.rotation.x,  0.1); rArm.rotation.y  = L(rArm.rotation.y,  0); rArm.rotation.z  = L(rArm.rotation.z, -1.5); }
  if (lFore) { lFore.rotation.x = 0; lFore.rotation.y = L(lFore.rotation.y, -0.1); lFore.rotation.z = 0; }
  if (rFore) { rFore.rotation.x = 0; rFore.rotation.y = L(rFore.rotation.y,  0.1); rFore.rotation.z = 0; }
  if (chest) { chest.rotation.x = L(chest.rotation.x, breathe); }
  if (spine) { spine.rotation.x = L(spine.rotation.x, breathe * 0.5); }
}

// ── VRM 아바타 컴포넌트 ───────────────────────────────────────
interface VRMAvatarProps {
  clip: MotionClip | null;
  playing: boolean;
  frozen: boolean;
  avatarUrl: string;
}

function VRMAvatar({ clip, playing, frozen, avatarUrl }: VRMAvatarProps) {
  const [vrm, setVrm] = useState<VRM | null>(null);
  const elapsedRef   = useRef(0);
  const prevGlossRef = useRef<string | null>(null);
  const frozenRef    = useRef(frozen);
  const playingRef   = useRef(playing);
  const clipRef      = useRef(clip);
  frozenRef.current  = frozen;
  playingRef.current = playing;
  clipRef.current    = clip;

  useEffect(() => {
    setVrm(null);
    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    loader.load(avatarUrl, gltf => {
      const v = gltf.userData.vrm as VRM | undefined;
      if (!v) return;
      VRMUtils.removeUnnecessaryVertices(v.scene);
      VRMUtils.combineSkeletons(v.scene);
      v.scene.traverse(obj => { obj.frustumCulled = false; });

      // head bone 기준으로 Y 오프셋 계산 (키 무관하게 상반신만 표시)
      v.scene.updateWorldMatrix(true, true);
      const headNode = v.humanoid.getRawBoneNode(VRMHumanBoneName.Head);
      if (headNode) {
        const headPos = new THREE.Vector3();
        headNode.getWorldPosition(headPos);
        v.scene.position.y = 0.5 - headPos.y;
      } else {
        const box = new THREE.Box3().setFromObject(v.scene);
        v.scene.position.y = 0.5 - box.max.y;
      }
      // chibi 모델처럼 실제 머리 메시가 head bone보다 큰 경우 카메라 상한(y≈0.7) 보정
      v.scene.updateWorldMatrix(true, true);
      const box = new THREE.Box3().setFromObject(v.scene);
      if (box.max.y > 0.7) {
        v.scene.position.y -= (box.max.y - 0.7);
      }

      setVrm(v);
    });
  }, [avatarUrl]);

  useFrame((_, delta) => {
    if (!vrm) return;

    const playing = playingRef.current;
    const frozen  = frozenRef.current;
    const clip    = clipRef.current;

    const hum  = vrm.humanoid;
    const expr = vrm.expressionManager;
    const sp   = playing ? 0.35 : 0.08;
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

    // ── 애니메이션 재생 ───────────────────────────────────────
    const animData = clip?.animation_data;
    const frames   = clip?.keypoints;
    const fps      = clip?.fps ?? 15;

    const hasAnim = !!(animData || (frames && frames.length > 0));

    if (playing && !frozen && hasAnim) {
      elapsedRef.current += delta;

      if (animData) {
        driveAnimData(animData, elapsedRef.current, hum, sp);
      } else if (frames && frames.length > 0) {
        const totalFrames = frames.length;
        const framePos    = (elapsedRef.current * fps) % totalFrames;
        const f0 = Math.floor(framePos);
        const f1 = Math.min(f0 + 1, totalFrames - 1);
        const t  = framePos - f0;
        const kp = lerpFrames(frames[f0], frames[f1], t);
        drivePoseBones(kp, hum, sp);
        driveHandBones(kp, hum, sp);
      }
    } else if (!frozen || !hasAnim) {
      // 대기 자세: frozen이어도 재생할 애니메이션이 없으면 idle 유지 (T포즈 방지)
      driveIdlePose(hum, playing ? sp : idle);
    }
    // frozen && hasAnim → 마지막 프레임 포즈 그대로 유지

    vrm.update(delta);
  });

  if (!vrm) return null;
  return (
    <primitive
      object={vrm.scene}
      rotation={[0, Math.PI, 0]}
    />
  );
}

// ── 메인 AvatarScene ─────────────────────────────────────────
interface AvatarSceneProps {
  clip: MotionClip | null;
  status: string;
  currentIndex?: number;
  total?: number;
  frozen?: boolean;
  avatarUrl?: string;
}

export default function AvatarScene({
  clip,
  status,
  frozen = false,
  avatarUrl = "/avatar.glb",
}: AvatarSceneProps) {
  const playing = status === "streaming";

  return (
    <div className="avatar-scene-wrap">
      <Canvas
        camera={{ position: [0, 0.2, 0.95], fov: 58 }}
        shadows
        gl={{ alpha: true }}
        onCreated={({ gl, scene }) => {
          gl.setClearColor(0x000000, 0);
          scene.background = null;
        }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[1, 3, 2]} intensity={1.4} castShadow />
        <pointLight position={[-1, 2, 2]} intensity={0.5} color="#c4b5fd" />

        <VRMAvatar clip={clip} playing={playing} frozen={frozen} avatarUrl={avatarUrl} />
        <Environment preset="city" />

        <OrbitControls
          enablePan={false}
          enableRotate={false}
          enableZoom={false}
          target={[0, 0.15, 0]}
        />
      </Canvas>
    </div>
  );
}
