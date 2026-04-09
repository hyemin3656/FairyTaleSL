/**
 * AvatarScene — React Three Fiber 기반 3D 아바타 씬
 *
 * 더미 GLTF 단계에서는 절차적 인체형 메시를 사용.
 * 실제 아바타 GLB가 준비되면 <useGLTF> 로 교체 예정.
 *
 * 감정(emotion_label)에 따라 몸통 색상 변경,
 * 클립 전환 시 바운스 애니메이션 재생.
 */
import { useEffect, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import type { MotionClip } from "../../api/gloss";

// ── 감정 → 색상 매핑 ──────────────────────────────────────────
const EMOTION_COLORS: Record<string, string> = {
  neutral:   "#6366f1",   // 보라
  happy:     "#f59e0b",   // 노랑
  sad:       "#60a5fa",   // 파랑
  angry:     "#ef4444",   // 빨강
  surprised: "#10b981",   // 초록
};

function emotionColor(label: string): string {
  return EMOTION_COLORS[label] ?? EMOTION_COLORS.neutral;
}

// ── 절차적 인체형 아바타 ──────────────────────────────────────
interface AvatarMeshProps {
  clip: MotionClip | null;
  playing: boolean;
}

function AvatarMesh({ clip, playing }: AvatarMeshProps) {
  const groupRef  = useRef<THREE.Group>(null!);
  const bodyRef   = useRef<THREE.Mesh>(null!);
  const headRef   = useRef<THREE.Mesh>(null!);
  const lArmRef   = useRef<THREE.Mesh>(null!);
  const rArmRef   = useRef<THREE.Mesh>(null!);

  // 클립 바뀔 때 bounce 트리거
  const bounceRef = useRef(0);
  useEffect(() => {
    if (clip) bounceRef.current = 1;
  }, [clip]);

  const color = emotionColor(clip?.emotion_label ?? "neutral");
  const matColor = new THREE.Color(color);

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // 상하 부유
    groupRef.current.position.y =
      Math.sin(Date.now() * 0.002) * 0.05;

    // 바운스 애니메이션
    if (bounceRef.current > 0) {
      bounceRef.current = Math.max(0, bounceRef.current - delta * 3);
      const scale = 1 + Math.sin(bounceRef.current * Math.PI) * 0.12;
      groupRef.current.scale.setScalar(scale);
    } else {
      groupRef.current.scale.setScalar(1);
    }

    // 팔 흔들기 (재생 중)
    if (playing) {
      const swing = Math.sin(Date.now() * 0.004) * 0.6;
      if (lArmRef.current) lArmRef.current.rotation.z =  swing + 0.3;
      if (rArmRef.current) rArmRef.current.rotation.z = -swing - 0.3;
    } else {
      if (lArmRef.current) lArmRef.current.rotation.z =  0.3;
      if (rArmRef.current) rArmRef.current.rotation.z = -0.3;
    }

    // mouthSmile → 머리 스케일 Y 미세 변화 (블렌드셰이프 시뮬)
    const smile = clip?.blendshape_params?.mouthSmile ?? 0;
    if (headRef.current) {
      headRef.current.scale.y = 1 + smile * 0.08;
    }
  });

  return (
    <group ref={groupRef}>
      {/* 몸통 */}
      <mesh ref={bodyRef} position={[0, 0, 0]} castShadow>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color={matColor} roughness={0.4} metalness={0.1} />
      </mesh>

      {/* 머리 */}
      <mesh ref={headRef} position={[0, 0.65, 0]} castShadow>
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial color={matColor} roughness={0.4} metalness={0.1} />
      </mesh>

      {/* 눈 (왼) */}
      <mesh position={[-0.08, 0.68, 0.2]}>
        <sphereGeometry args={[0.035, 16, 16]} />
        <meshStandardMaterial color="#1e1b4b" />
      </mesh>
      {/* 눈 (오) */}
      <mesh position={[0.08, 0.68, 0.2]}>
        <sphereGeometry args={[0.035, 16, 16]} />
        <meshStandardMaterial color="#1e1b4b" />
      </mesh>

      {/* 왼팔 */}
      <mesh ref={lArmRef} position={[-0.35, 0.05, 0]} rotation={[0, 0, 0.3]} castShadow>
        <capsuleGeometry args={[0.07, 0.45, 8, 16]} />
        <meshStandardMaterial color={matColor} roughness={0.5} />
      </mesh>

      {/* 오른팔 */}
      <mesh ref={rArmRef} position={[0.35, 0.05, 0]} rotation={[0, 0, -0.3]} castShadow>
        <capsuleGeometry args={[0.07, 0.45, 8, 16]} />
        <meshStandardMaterial color={matColor} roughness={0.5} />
      </mesh>

      {/* 왼다리 */}
      <mesh position={[-0.14, -0.6, 0]} castShadow>
        <capsuleGeometry args={[0.08, 0.4, 8, 16]} />
        <meshStandardMaterial color={matColor} roughness={0.5} />
      </mesh>

      {/* 오른다리 */}
      <mesh position={[0.14, -0.6, 0]} castShadow>
        <capsuleGeometry args={[0.08, 0.4, 8, 16]} />
        <meshStandardMaterial color={matColor} roughness={0.5} />
      </mesh>
    </group>
  );
}

// ── 글로스 자막 오버레이 (DOM) ────────────────────────────────
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
      {/* Three.js 캔버스 */}
      <Canvas
        camera={{ position: [0, 0.3, 2.2], fov: 45 }}
        shadows
        style={{ background: "#0f0e1a", borderRadius: 12 }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[3, 5, 3]}
          intensity={1.2}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <pointLight position={[-2, 2, -2]} intensity={0.4} color="#a5b4fc" />

        <AvatarMesh clip={clip} playing={playing} />

        <ContactShadows
          position={[0, -1.05, 0]}
          opacity={0.4}
          scale={2}
          blur={1.5}
        />
        <Environment preset="city" />
        <OrbitControls
          enablePan={false}
          minDistance={1.5}
          maxDistance={4}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI / 1.8}
        />
      </Canvas>

      {/* 글로스 오버레이 (캔버스 위) */}
      <GlossOverlay
        clip={clip}
        status={status}
        currentIndex={currentIndex}
        total={total}
      />

      {/* 글로스 토큰 바 */}
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
