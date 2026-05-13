import http from "./http";

export interface MotionClip {
  gloss: string;
  gltf_clip_url: string;
  emotion_label: string;
  blendshape_params: Record<string, number>;
  duration_sec: number;
  is_fallback: boolean;
  fallback_type?: string | null;
  text_display?: string | null;
  // N×225 keypoint 시퀀스 (15fps): lhand(0..62) + rhand(63..125) + pose(126..224)
  keypoints?: number[][] | null;
  fps?: number;
  // 사전 계산된 본 회전 데이터 (step6_bake_anim.py 생성)
  animation_data?: AnimData | null;
}

export interface AnimData {
  fps: number;
  n: number;
  bones: Record<string, number[] | [number, number, number][]>;
}

export interface GlossConvertResponse {
  original_text: string;
  tokens: string[];
  clips: MotionClip[];
  total_duration_sec: number;
}

export async function convertTextToGloss(
  text: string
): Promise<GlossConvertResponse> {
  const { data } = await http.post<GlossConvertResponse>("/gloss/convert", {
    text,
  });
  return data;
}
