import http from "./http";

export interface MotionClip {
  gloss: string;
  gltf_clip_url: string;
  emotion_label: string;
  blendshape_params: Record<string, number>;
  duration_sec: number;
  is_fallback: boolean;
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
